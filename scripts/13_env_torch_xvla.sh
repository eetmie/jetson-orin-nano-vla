#!/usr/bin/env bash
# venv for the X-VLA PyTorch baseline. Separate from .venv-torch because lerobot 0.5.1
# ships a different X-VLA/Florence2 layout than the export tooling needs.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${1:-.venv-torch-xvla}"
INDEX="https://pypi.jetson-ai-lab.io/sbsa/cu130"

python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -U pip wheel
"$VENV/bin/pip" install -r requirements/torch-xvla.txt
# PINNED, not latest. An unpinned resolve picks up torch 2.13.0, whose
# libtorch_cuda.so wants `ncclCommResume` -- a symbol JetPack 7.2 does not provide, so
# `import torch` dies with an ImportError that names the symbol and nothing else. These
# are the wheels .venv-torch already runs.
"$VENV/bin/pip" install --force-reinstall --no-deps \
    "torch==2.11.0" "torchvision==0.26.0" --extra-index-url "$INDEX"
# See requirements/torch-xvla.txt: the JetPack scipy shadows through and breaks
# `import lerobot`. Install both into the venv rather than letting pip downgrade numpy.
"$VENV/bin/pip" install --ignore-installed "numpy>=2.2.6" "scipy>=1.14"

"$VENV/bin/python" - <<'PY'
import torch, numpy, scipy
print("torch", torch.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__)
assert torch.cuda.is_available(), (
    "torch cannot see the GPU — a resolver almost certainly replaced the "
    "JetPack-matched wheel. Reinstall from "
    "https://pypi.jetson-ai-lab.io/sbsa/cu130 and re-run.")
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy   # noqa: F401
print("lerobot xvla policy importable, device", torch.cuda.get_device_name(0))
PY
echo "OK -> $VENV"
