#!/usr/bin/env bash
# venv for the stock-PyTorch baseline.
#
# The trap this script exists to avoid: `pip install lerobot` pulls torch from PyPI,
# and the aarch64 PyPI wheel is CPU-only. Install it after the JetPack-matched wheel
# and the "GPU baseline" silently becomes a CPU run that is ~20x slower — a wrong
# number rather than an error. So: lerobot first, JetPack torch forced on top, then
# an assert that CUDA is actually there.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${1:-.venv-torch}"
INDEX="https://pypi.jetson-ai-lab.io/sbsa/cu130"   # CUDA 13 aarch64. There is no jp7 index.

python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip wheel
"$VENV/bin/pip" install -r requirements/torch.txt
"$VENV/bin/pip" install --force-reinstall --no-deps torch torchvision --extra-index-url "$INDEX"

"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), (
    "torch cannot see the GPU. The PyPI aarch64 wheel is CPU-only — reinstall from "
    "https://pypi.jetson-ai-lab.io/sbsa/cu130 and re-run.")
print("device", torch.cuda.get_device_name(0),
      "capability", torch.cuda.get_device_capability(0))
PY
echo "OK -> $VENV"
