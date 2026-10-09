#!/usr/bin/env bash
# venv for the stock-PyTorch baseline.
#
# The trap this script exists to avoid: `pip install lerobot` pulls torch from PyPI,
# which will happily replace the JetPack-matched wheel. Install it in that order
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
# PINNED, not latest: an unpinned resolve picks up torch 2.13.0, whose libtorch_cuda.so
# wants `ncclCommResume`, a symbol JetPack 7.2 does not provide. Same pin as
# 13_env_torch_xvla.sh.
"$VENV/bin/pip" install --force-reinstall --no-deps \
    "torch==2.11.0" "torchvision==0.26.0" --extra-index-url "$INDEX"
# --no-deps leaves out the cu13 runtime wheels this torch links against (cusparseLt,
# NCCL, ...): lerobot 0.5.1 resolved PyPI torch 2.10, which does not pull them. A
# plain install of the same pin keeps the wheel above and adds only what it needs.
"$VENV/bin/pip" install "torch==2.11.0" "torchvision==0.26.0" --extra-index-url "$INDEX"

"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), (
    "torch cannot see the GPU — a resolver almost certainly replaced the "
    "JetPack-matched wheel. Reinstall from "
    "https://pypi.jetson-ai-lab.io/sbsa/cu130 and re-run.")
print("device", torch.cuda.get_device_name(0),
      "capability", torch.cuda.get_device_capability(0))
PY
echo "OK -> $VENV"
