#!/usr/bin/env bash
# Build the exporter venvs on the machine you fine-tune on (not on the Jetson).
#
#   export/setup.sh            # both
#   export/setup.sh smolvla    # or just one
#
# Two venvs because lerobot 0.5.1 (SmolVLA) and 0.6.1 (X-VLA) cannot share one.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FAMS=("$@"); [[ ${#FAMS[@]} -gt 0 ]] || FAMS=(smolvla xvla)
for f in "${FAMS[@]}"; do
  venv="$HERE/.venv-$f"
  python3 -m venv "$venv"
  "$venv/bin/pip" install -U pip wheel
  "$venv/bin/pip" install -r "$HERE/requirements-$f.txt"
  "$venv/bin/python" -c "import torch, lerobot, onnx; print('$f:', 'torch', torch.__version__, '| lerobot', lerobot.__version__, '| onnx', onnx.__version__)"
done
