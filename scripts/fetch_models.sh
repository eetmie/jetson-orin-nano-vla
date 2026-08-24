#!/usr/bin/env bash
# Pull model artefacts from the Hub.
#
#   scripts/fetch_models.sh smolvla-base         # torch weights + the split ONNX
#   scripts/fetch_models.sh xvla-base            # torch weights (no ONNX published)
#
# Uses the `hf` CLI rather than snapshot_download: large safetensors have stalled here
# — process alive, file not growing, no exception raised, so the library's own retry
# never fires. If `hf` stalls too, the curl form at the bottom turns a stall into an
# error that --retry can act on, and -C - resumes.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${DEST:-$HOME/bundles}"
MODEL="${1:?usage: fetch_models.sh <smolvla-base|xvla-base>}"

case "$MODEL" in
  smolvla-base)
    TORCH_REPO="lerobot/smolvla_base"
    SPLIT_REPO="ainekko/smolvla_base_onnx"   # the only public split SmolVLA ONNX
    ;;
  xvla-base)
    TORCH_REPO="lerobot/xvla-base"
    SPLIT_REPO=""                            # none published — export it, see docs/03
    ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac

mkdir -p "$DEST"
echo ">> $TORCH_REPO -> $DEST/$MODEL-torch"
hf download "$TORCH_REPO" --local-dir "$DEST/$MODEL-torch"

if [[ -n "$SPLIT_REPO" ]]; then
    echo ">> $SPLIT_REPO -> $DEST/$MODEL-split"
    hf download "$SPLIT_REPO" --local-dir "$DEST/$MODEL-split"
else
    echo ">> no published split ONNX for $MODEL — export it (docs/03-backends.md)"
fi

echo
echo "done. Point the benchmark at them:"
echo "  python -m bench torch     --model $MODEL --checkpoint $DEST/$MODEL-torch"
[[ -n "$SPLIT_REPO" ]] && \
echo "  python -m bench ort-split --model $MODEL --bundle     $DEST/$MODEL-split"

# Fallback for a stalling download of one big file:
#   curl -L --retry 10 --retry-delay 5 --speed-limit 1000000 --speed-time 30 -C - \
#     -o model.safetensors \
#     "https://huggingface.co/<repo>/resolve/main/model.safetensors"
