#!/usr/bin/env bash
# The supported benchmark path for one model, in the order that fails cheapest first.
#
#   MODEL=smolvla-base scripts/run_all.sh
#   MODEL=xvla-base    scripts/run_all.sh
#
# VIEWS must match the selected export bundle. Camera count is static for X-VLA and
# camera slots change SmolVLA sequence shape, so cross-view sweeps need separate bundles.
#
# Each backend runs in its own venv (they cannot share one — see docs/02). Every run
# writes results/<label>.json and the script keeps going if one fails, because "this
# backend does not work on this board" is a result worth having written down.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-smolvla-base}"
DEST="${DEST:-$HOME/bundles}"
CKPT="${CKPT:-$DEST/$MODEL-torch}"
BUNDLE="${BUNDLE:-$DEST/$MODEL-split}"
ITERS="${ITERS:-100}"
OBS="${OBS:-synthetic}"
case "$MODEL" in
    smolvla-base) MODEL_FAMILY=smolvla ;;
    xvla-base)    MODEL_FAMILY=xvla ;;
    *) echo "MODEL must be smolvla-base or xvla-base"; exit 2 ;;
esac
VIEWS="${VIEWS:-}"
if [[ -z "$VIEWS" ]]; then
    [[ "$MODEL_FAMILY" == "xvla" ]] && VIEWS=3 || VIEWS=2
fi

MODEL_ARGS=(--model "$MODEL")
[[ -n "${TASK:-}"       ]] && MODEL_ARGS+=(--task       "$TASK")

COMMON=(--iters "$ITERS" --obs "$OBS" --idle-s 5 --warmup 5 "${MODEL_ARGS[@]}")

if [[ "$MODEL_FAMILY" == "xvla" ]]; then
    VENV_TORCH="${VENV_TORCH:-.venv-torch-xvla}"
else
    VENV_TORCH="${VENV_TORCH:-.venv-torch}"
fi
VENV_ORT="${VENV_ORT:-.venv-ort}"

mkdir -p results
run() {  # run <venv> <label> <bench-args...>
    local venv="$1" label="$2"; shift 2
    if [[ ! -x "$venv/bin/python" ]]; then
        echo "!! skipping $label: $venv missing (run the matching scripts/1*_env_*.sh)"
        return
    fi
    echo; echo "───────── $label ─────────"
    "$venv/bin/python" -m bench "$@" --label "$label" || echo "!! $label failed (recorded)"
}

echo "== board state =="
scripts/00_host_prep.sh --verify | head -20

# 1. PyTorch first: no export, no engine build. If this fails, the problem is the
#    environment rather than any backend.
TORCH_BUNDLE_ARGS=()
[[ -d "$BUNDLE" ]] && TORCH_BUNDLE_ARGS+=(--bundle "$BUNDLE")
run "$VENV_TORCH" "$MODEL.torch-fp32" torch --checkpoint "$CKPT" \
    "${TORCH_BUNDLE_ARGS[@]}" "${COMMON[@]}"

# 2. The split path. First run builds every engine, one subprocess per graph — ~5 min
#    for SmolVLA, ~10 for X-VLA. Later runs load from cache in seconds.
if [[ -d "$BUNDLE" ]]; then
    for v in $VIEWS; do
        run "$VENV_ORT" "$MODEL.ort-split.${v}cam" ort-split --bundle "$BUNDLE" \
            --views "$v" "${COMMON[@]}"
    done
else
    echo "!! no split bundle at $BUNDLE — run scripts/fetch_models.sh $MODEL"
fi

# 3. Sustained run, to catch thermal drift the short runs miss.
if [[ "${SUSTAINED:-1}" == "1" && -d "$BUNDLE" ]]; then
    run "$VENV_ORT" "$MODEL.ort-split.sustained" ort-split --bundle "$BUNDLE" \
        --duration-s "${SUSTAINED_S:-300}" \
        --obs "$OBS" --idle-s 5 --warmup 5 "${MODEL_ARGS[@]}"
fi

echo; echo "== report =="
"${VENV_TORCH}/bin/python" -m bench report results \
    --out docs/RESULTS.md 2>/dev/null \
  || python3 -m bench report results \
    --out docs/RESULTS.md
echo "wrote docs/RESULTS.md"
