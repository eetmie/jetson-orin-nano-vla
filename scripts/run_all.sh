#!/usr/bin/env bash
# The whole comparison, in the order that fails cheapest first.
#
#   BUNDLE=~/bundles/smolvla-digging-clean-ir12-35k \
#   CKPT=~/bundles/clean_ir12-035000/pretrained_model \
#   scripts/run_all.sh
#
# Each backend runs in its own venv (they cannot share one — see docs/02). Every run
# writes results/<label>.json and keeps going if one fails, because "this backend
# does not work on this board" is a result worth having written down.
set -uo pipefail
cd "$(dirname "$0")/.."

BUNDLE="${BUNDLE:?set BUNDLE to the split export bundle directory}"
CKPT="${CKPT:?set CKPT to the LeRobot pretrained_model directory}"
ITERS="${ITERS:-100}"
OBS="${OBS:-synthetic}"
COMMON=(--iters "$ITERS" --obs "$OBS" --idle-s 5 --warmup 5)

VENV_TORCH="${VENV_TORCH:-.venv-torch}"
VENV_ORT="${VENV_ORT:-.venv-ort}"
VENV_TETHER="${VENV_TETHER:-.venv-tether}"

mkdir -p results
run() {  # run <venv> <label> <args...>
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

# 1. PyTorch first: no export step, no engine build, so if this fails everything else
#    is downstream of a broken environment rather than a broken backend.
run "$VENV_TORCH" torch-fp32   torch --checkpoint "$CKPT" --bundle "$BUNDLE" \
    --weights float32 --autocast off "${COMMON[@]}"
run "$VENV_TORCH" torch-amp16  torch --checkpoint "$CKPT" --bundle "$BUNDLE" \
    --weights float32 --autocast float16 "${COMMON[@]}"
run "$VENV_TORCH" torch-half16 torch --checkpoint "$CKPT" --bundle "$BUNDLE" \
    --weights float16 --autocast off --patch-half-out "${COMMON[@]}"

# 2. The incumbent. The first run builds three TensorRT engines (~5 min, one
#    subprocess per graph — two builds in one process OOM 8 GB). Later runs load
#    from ~/.cache/jetson-orin-nano-vla/trt in seconds.
run "$VENV_ORT" ort-split-fp16 ort-split --bundle "$BUNDLE" --precision fp16 "${COMMON[@]}"

# 3. Tether. Give it a long startup budget: a monolithic TensorRT build on this board
#    is minutes at best, and the interesting outcome may be that it never finishes.
run "$VENV_TETHER" tether-trt tether --export-dir "${TETHER_EXPORT:-$BUNDLE}" \
    --bundle "$BUNDLE" --providers TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider \
    --startup-timeout 1800 "${COMMON[@]}"

# 4. Sustained run on whatever won, to catch thermal drift the short runs miss.
if [[ "${SUSTAINED:-1}" == "1" ]]; then
    run "$VENV_ORT" ort-split-fp16-sustained ort-split --bundle "$BUNDLE" \
        --precision fp16 --duration-s "${SUSTAINED_S:-300}" --obs "$OBS" --idle-s 5 --warmup 5
fi

echo; echo "== report =="
"${VENV_TORCH}/bin/python" -m bench report results --out docs/RESULTS.md 2>/dev/null \
  || python3 -m bench report results --out docs/RESULTS.md
echo "wrote docs/RESULTS.md"
