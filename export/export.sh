#!/usr/bin/env bash
# Export a SmolVLA or X-VLA checkpoint to the split ONNX bundle the benchmark runs.
# Run it where you fine-tune, then copy the bundle to the Jetson.
#
#   export/export.sh <checkpoint dir | HF id> <out dir> [--views N] [--task "..."]... [--fps N]
#   export/export.sh nvidia/GR00T-N1.6-3B <out dir> [--embodiment E] [--views N] [--task "..."]
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
VIEWS=""; EXTRA=(); EMBODIMENT=robocasa_panda_omron; TASKS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --views) VIEWS="$2"; shift 2 ;;
    --embodiment) EMBODIMENT="$2"; shift 2 ;;
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
    PYTHONPATH="$HERE:$HERE/smolvla" "$V/bin/python" "$HERE/smolvla/export_split_onnx.py" \
        --model-id "$CKPT" --out-dir "$OUT" --views "$VIEWS" "${EXTRA[@]}"
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
    ;;
  groot)
    # Not a LeRobot policy: NVIDIA's model code (export/setup.sh groot) loads it, the
    # stock model writes the parity fixture, then the split + mixed-FP16 pass runs.
    V=$(venv_for groot)
    export PYTHONPATH="$HERE:$HERE/groot"
    [[ ${#TASKS[@]} -le 1 ]] || { echo "GR00T bakes one prompt into the bundle: pass one --task"; exit 1; }
    TASK_ARGS=(); [[ ${#TASKS[@]} -eq 1 ]] && TASK_ARGS=(--task "${TASKS[0]}")
    FIXTURE="$OUT.fixture.npz"
    "$V/bin/python" "$HERE/groot/reference.py" --checkpoint "$CKPT" --embodiment "$EMBODIMENT" \
        --views "$VIEWS" "${TASK_ARGS[@]}" --out "$FIXTURE"
    "$V/bin/python" "$HERE/groot/export_split_onnx.py" --ref "$FIXTURE" --out "$OUT"
    rm -f "$FIXTURE"
    ;;
  *) echo "unsupported policy type '$FAMILY' (supported: smolvla, xvla, GR00T N1.6)"; exit 1 ;;
esac

( cd "$OUT" && sha256sum --quiet -c MANIFEST.sha256 ) && echo ">> manifest OK"
echo
echo "Copy $OUT to the Jetson, then benchmark it there:"
if [[ "$FAMILY" == groot ]]; then
  echo "  .venv-ort/bin/python -m bench trt-split --model groot-n16-base --bundle <bundle>"
else
  echo "  .venv-ort/bin/python -m bench ort-split --model ${FAMILY}-base --bundle <bundle> --views $VIEWS"
fi
