#!/usr/bin/env bash
# Export lerobot/xvla-base as split ONNX graphs. RUN THIS OFF THE ORIN.
#
# Loading the policy costs ~3.5 GB on CPU and the exporter holds that alongside an
# export trace, one subprocess per graph. It fits on a workstation; on an 8 GB board it
# swaps. Export elsewhere, rsync the graphs over, build the engines on the target — a
# TensorRT engine is hardware- and version-specific and is never copied.
#
#   CHECKPOINT=~/models/xvla-base OUT=~/bundles/xvla-base-split scripts/export_xvla_split.sh
#
# The exporter itself lives in the X-VLA runtime project, not here:
#   spark-projects/orin-nano/xvla-runtime/tools/export_split_onnx.py
# Point EXPORTER at it, or set it to your own copy.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the xvla-base directory}"
OUT="${OUT:-$HOME/bundles/xvla-base-split}"
EXPORTER="${EXPORTER:-$HOME/spark-projects/orin-nano/xvla-runtime/tools/export_split_onnx.py}"
VALID_VIEWS="${VALID_VIEWS:-1}"
LANG_LEN="${LANG_LEN:-50}"
DOMAIN_ID="${DOMAIN_ID:-0}"
BUDGET_GB="${BUDGET_GB:-0.40}"
PY="${PY:-python3}"

# lerobot version matters here, and not in the usual way. The exporter reaches into
# lerobot's VENDORED Florence2 (`lerobot/policies/xvla/modeling_florence2.py`), whose
# module layout has changed between releases: some revisions expose
# `vlm.multi_modal_projector` and DaViT conv/block modules taking a bare tensor; others
# fold the projector into `image_projection` + `image_proj_norm` and pass
# `(x, input_size)` pairs. A mismatch fails at export with an AttributeError or a
# tracing error, not with anything that names the real cause. Export with the same
# lerobot the exporter was written against — 0.6.1 — in its own venv.
$PY -c "import lerobot, sys; v=lerobot.__version__; print('lerobot', v);
sys.exit(0 if v.startswith('0.6') else 1)" || {
    echo "!! this interpreter's lerobot is not 0.6.x."
    echo "   python3 -m venv .venv-xvla-export"
    echo "   .venv-xvla-export/bin/pip install 'lerobot[xvla]==0.6.1' onnx"
    echo "   PY=.venv-xvla-export/bin/python scripts/export_xvla_split.sh"
    exit 1
}

echo ">> exporting $CHECKPOINT -> $OUT (valid_views=$VALID_VIEWS, budget ${BUDGET_GB} GB)"
$PY "$EXPORTER" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUT" \
    --domain-id "$DOMAIN_ID" \
    --valid-views "$VALID_VIEWS" \
    --lang-len "$LANG_LEN" \
    --budget-gb "$BUDGET_GB"

echo
echo ">> graphs:"
ls -la "$OUT"
echo
echo "Next: parity-check against the PyTorch reference BEFORE trusting any action,"
echo "then rsync to the board and build the engines there:"
echo "  python -m bench ort-split --model xvla-base --bundle $OUT --precision fp16"
echo
echo "To publish it: hf/xvla-base-onnx-README.md is a model card ready to go with it."
