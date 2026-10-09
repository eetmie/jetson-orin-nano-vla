#!/usr/bin/env bash
# Export a SmolVLA, X-VLA or EVO1 checkpoint to the split ONNX bundle the benchmark runs.
# Run it where you fine-tune, then copy the bundle to the Jetson.
#
#   export/export.sh <checkpoint dir | HF id> <out dir> [--views N] [--task "..."]... [--fps N]
#   export/export.sh nvidia/GR00T-N1.6-3B <out dir> [--embodiment E] [--views N] [--task "..."]
#   export/export.sh nvidia/GR00T-N1.7-3B <out dir> [--embodiment E] [--task "..."] [--vlm-files D]
#
# GR00T N1.7 takes its tokenizer/processor from nvidia/Cosmos-Reason2-2B, a gated
# Hugging Face repo: accept its terms first, or pass --vlm-files with a local copy.
#
# The family comes from the checkpoint's config.json. --views defaults to the number of
# observation.images.* inputs the checkpoint declares; it is baked into the graphs, so a
# bundle only serves runtimes feeding that many cameras or fewer.
#
# SmolVLA bundles stay FP32 (TensorRT builds FP16 engines on the board). X-VLA bundles get
# the mixed-FP16 weight pass, which halves what stays resident on an 8 GB board.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="${1:?usage: export.sh <checkpoint dir | HF id> <out dir> [--views N] [--task T]... [--fps N]}"
OUT="${2:?usage: export.sh <checkpoint dir | HF id> <out dir> [--views N] [--task T]... [--fps N]}"
shift 2
VIEWS=""; EXTRA=(); EMBODIMENT=""; TASKS=(); VLM_FILES=nvidia/Cosmos-Reason2-2B
while [[ $# -gt 0 ]]; do
  case "$1" in
    --views) VIEWS="$2"; shift 2 ;;
    --embodiment) EMBODIMENT="$2"; shift 2 ;;
    --vlm-files) VLM_FILES="$2"; shift 2 ;;
    --task) EXTRA+=("$1" "$2"); TASKS+=("$2"); shift 2 ;;
    --fps) EXTRA+=("$1" "$2"); shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

venv_for() { [[ -x "$HERE/.venv-$1/bin/python" ]] || { echo "missing $HERE/.venv-$1: run export/setup.sh $1"; exit 1; }; echo "$HERE/.venv-$1"; }

if [[ -d "$SRC" ]]; then
  CKPT=$(cd "$SRC" && pwd)
else
  CKPT="$HERE/.checkpoints/${SRC//\//--}"
  HF=$(ls "$HERE"/.venv-*/bin/hf 2>/dev/null | head -1)
  [[ -n "$HF" ]] || { echo "no exporter venv yet: run export/setup.sh first"; exit 1; }
  "$HF" download "$SRC" --local-dir "$CKPT"
fi
[[ -f "$CKPT/config.json" ]] || { echo "$CKPT has no config.json: not a LeRobot checkpoint"; exit 1; }

read -r FAMILY NIMG < <(python3 - "$CKPT/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
if c.get("model_type") == "Gr00tN1d6":
    print("groot", 3)        # robocasa_panda_omron; pass --views for another embodiment
elif c.get("model_type") == "Gr00tN1d7":
    print("groot17", 3)      # xdof_relative_eef_relative_joint; views come from the embodiment
else:
    print(c.get("type"), sum(k.startswith("observation.images.") for k in c.get("input_features", {})))
PY
)
VIEWS="${VIEWS:-$NIMG}"
[[ "$VIEWS" -gt 0 ]] || { echo "the checkpoint declares no image inputs: pass --views"; exit 1; }
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
echo ">> $FAMILY checkpoint $CKPT -> $OUT  (views $VIEWS)"

case "$FAMILY" in
  smolvla)
    V=$(venv_for smolvla)
    export PYTHONPATH="$HERE:$HERE/smolvla"
    "$V/bin/python" "$HERE/smolvla/export_split_onnx.py" \
        --model-id "$CKPT" --out-dir "$OUT" --views "$VIEWS" "${EXTRA[@]}"
    # Mixed FP16 for the three heavy graphs (RMSNorms, LayerNorm, Softmax stay FP32),
    # then the stock policy's chunk for seeded inputs and the token-embedding table.
    "$V/bin/python" -m vla_common.fp16_mixed --bundle "$OUT" --graphs \
        smolvlm_vision.onnx smolvlm_expert_prefill.onnx smolvlm_expert_decode.onnx
    TASK_ARGS=(); [[ ${#TASKS[@]} -ge 1 ]] && TASK_ARGS=(--task "${TASKS[0]}")
    "$V/bin/python" "$HERE/smolvla/reference.py" --checkpoint "$CKPT" --bundle "$OUT" "${TASK_ARGS[@]}"
    ;;
  xvla)
    V=$(venv_for xvla)
    export PYTHONPATH="$HERE:$HERE/xvla"
    FP32="$OUT.fp32"
    rm -rf "$FP32"
    "$V/bin/python" "$HERE/xvla/export_split_onnx.py" \
        --checkpoint "$CKPT" --out-dir "$FP32" --views "$VIEWS" "${EXTRA[@]}"
    "$V/bin/python" -m vla_common.fp16_weights --split-dir "$FP32" --out-dir "$OUT"
    cp "$FP32"/_meta_*.json "$OUT"/
    # Rewrite bundle.json and the manifest over the FP16 graphs.
    "$V/bin/python" "$HERE/xvla/export_split_onnx.py" \
        --checkpoint "$CKPT" --out-dir "$OUT" --bundle-only --views "$VIEWS" "${EXTRA[@]}"
    rm -rf "$FP32"
    # The stock policy's chunk for seeded inputs, which the board checks its engines against.
    TASK_ARGS=(); [[ ${#TASKS[@]} -ge 1 ]] && TASK_ARGS=(--task "${TASKS[0]}")
    "$V/bin/python" "$HERE/xvla/reference.py" --checkpoint "$CKPT" --bundle "$OUT" "${TASK_ARGS[@]}"
    ;;
  evo1)
    # LeRobot 0.6.1, the X-VLA venv. The VLM base is InternVL3-1B-hf at the revision the
    # LIBERO recipe trained from; the exporter refuses any other.
    V=$(venv_for xvla)
    export PYTHONPATH="$HERE:$HERE/evo1"
    BASE_REV=014c0583a0d4bedf29fbe2dbff4f865eb998e171
    BASE="$HERE/.checkpoints/OpenGVLab--InternVL3-1B-hf"
    if [[ "$(cat "$BASE/REVISION" 2>/dev/null)" != "$BASE_REV" ]]; then
      "$V/bin/hf" download OpenGVLab/InternVL3-1B-hf --revision "$BASE_REV" --local-dir "$BASE"
      echo "$BASE_REV" > "$BASE/REVISION"
    fi
    # Each view spends 256 image tokens whether or not a camera fills it; 64 are left for text.
    "$V/bin/python" "$HERE/evo1/export_split_onnx.py" --checkpoint "$CKPT" --base "$BASE" \
        --out-dir "$OUT" --views "$VIEWS" --seq-len $((VIEWS * 256 + 64))
    TASK_ARGS=(); [[ ${#TASKS[@]} -ge 1 ]] && TASK_ARGS=(--task "${TASKS[0]}")
    "$V/bin/python" "$HERE/evo1/reference.py" --bundle-dir "$OUT" --base "$BASE" "${TASK_ARGS[@]}"
    # Mixed FP16 for every engine graph (RMSNorms, LayerNorm, Softmax stay FP32).
    GRAPHS=$(cd "$OUT" && ls vision_*.onnx language_*.onnx action_*.onnx)
    "$V/bin/python" -m vla_common.fp16_mixed --bundle "$OUT" --graphs $GRAPHS
    ;;
  groot)
    # Not a LeRobot policy: NVIDIA's model code (export/setup.sh groot) loads it, the
    # stock model writes the parity fixture, then the split + mixed-FP16 pass runs.
    V=$(venv_for groot)
    export PYTHONPATH="$HERE:$HERE/groot"
    [[ ${#TASKS[@]} -le 1 ]] || { echo "GR00T bakes one prompt into the bundle: pass one --task"; exit 1; }
    TASK_ARGS=(); [[ ${#TASKS[@]} -eq 1 ]] && TASK_ARGS=(--task "${TASKS[0]}")
    FIXTURE="$OUT.fixture.npz"
    "$V/bin/python" "$HERE/groot/reference.py" --checkpoint "$CKPT" \
        --embodiment "${EMBODIMENT:-robocasa_panda_omron}" \
        --views "$VIEWS" "${TASK_ARGS[@]}" --out "$FIXTURE"
    "$V/bin/python" "$HERE/groot/export_split_onnx.py" --ref "$FIXTURE" --out "$OUT"
    rm -f "$FIXTURE"
    ;;
  groot17)
    # As N1.6, with the N1.7 model code (export/setup.sh groot17). The embodiment fixes
    # the camera count and the history frames; --views does not apply.
    V=$(venv_for groot17)
    export PYTHONPATH="$HERE:$HERE/groot"
    [[ ${#TASKS[@]} -le 1 ]] || { echo "GR00T bakes one prompt into the bundle: pass one --task"; exit 1; }
    TASK_ARGS=(); [[ ${#TASKS[@]} -eq 1 ]] && TASK_ARGS=(--task "${TASKS[0]}")
    FIXTURE="$OUT.fixture.npz"
    "$V/bin/python" "$HERE/groot/reference17.py" --checkpoint "$CKPT" \
        --embodiment "${EMBODIMENT:-xdof_relative_eef_relative_joint}" --vlm-files "$VLM_FILES" \
        "${TASK_ARGS[@]}" --out "$FIXTURE"
    "$V/bin/python" "$HERE/groot/export_split_onnx17.py" --ref "$FIXTURE" --out "$OUT"
    rm -f "$FIXTURE"
    ;;
  *) echo "unsupported policy type '$FAMILY' (supported: smolvla, xvla, evo1, GR00T N1.6, N1.7)"; exit 1 ;;
esac

( cd "$OUT" && sha256sum --quiet -c MANIFEST.sha256 ) && echo ">> manifest OK"
echo
echo "Copy $OUT to the Jetson, then benchmark it there:"
case "$FAMILY" in
  groot) PROFILE=groot-n16-base ;;
  groot17) PROFILE=groot-n17-base ;;
  evo1) PROFILE=evo1-libero ;;
  *) PROFILE=${FAMILY}-base ;;
esac
echo "  .venv-ort/bin/python -m bench trt-split --model $PROFILE --bundle <bundle>"
