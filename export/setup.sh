#!/usr/bin/env bash
# Build the exporter venvs on the machine you fine-tune on (not on the Jetson).
#
#   export/setup.sh            # smolvla and xvla
#   export/setup.sh smolvla    # or just one
#   export/setup.sh groot      # GR00T N1.6 (opt-in: ~6.6 GB checkpoint, own model code)
#   export/setup.sh groot17    # GR00T N1.7 (opt-in: ~6.9 GB checkpoint, own model code)
#
# One venv per family: lerobot 0.5.1 (SmolVLA), 0.6.1 (X-VLA), GR00T N1.6's
# transformers 4.51.3 and N1.7's 4.57.3 cannot share one.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FAMS=("$@"); [[ ${#FAMS[@]} -gt 0 ]] || FAMS=(smolvla xvla)
declare -A GROOT_REF=(
  [groot]=5dc80c4afd726b34faad1d8f7e007a13b34e4c88     # NVIDIA/Isaac-GR00T n1.6.1-release
  [groot17]=23ace64f17aa5015259b8609d371eb61a357c776   # NVIDIA/Isaac-GR00T n1.7-release
)
declare -A GROOT_SRC=([groot]=.isaac-gr00t [groot17]=.isaac-gr00t-n17)
for f in "${FAMS[@]}"; do
  venv="$HERE/.venv-$f"
  [[ ! -e "$venv" ]] || { echo "refusing existing environment $venv; choose a fresh checkout"; exit 1; }
  python3 -m venv "$venv"
  "$venv/bin/pip" install -c "$HERE/constraints-$f.txt" pip==26.2.1 wheel==0.48.0 setuptools
  # Exact dependency closure, including indirect imports needed by the stock policy.
  "$venv/bin/pip" install -r "$HERE/requirements-$f.txt"
  if [[ "$f" == groot* ]]; then
    ref=${GROOT_REF[$f]} src="$HERE/${GROOT_SRC[$f]}"
    if [[ "$(cat "$src/.source-commit" 2>/dev/null)" != "$ref" ]]; then
      [[ ! -e "$src" ]] || { echo "refusing an existing unverified source directory $src"; exit 1; }
      git init -q "$src"
      git -C "$src" remote add origin https://github.com/NVIDIA/Isaac-GR00T.git
      git -C "$src" sparse-checkout init --cone
      git -C "$src" sparse-checkout set gr00t
      git -C "$src" fetch --filter=blob:none --depth=1 origin "$ref"
      git -C "$src" checkout --detach FETCH_HEAD
      echo "$ref" > "$src/.source-commit"
    fi
    [[ "$(git -C "$src" rev-parse HEAD)" == "$ref" ]] || { echo "GR00T source checkout does not match $ref"; exit 1; }
    git -C "$src" diff --quiet HEAD -- || { echo "GR00T source has modified tracked files: $src"; exit 1; }
    # NVIDIA's dependency list pins flash-attn/deepspeed for training; the export needs
    # none. N1.7 declares python 3.10 only; the export runs on 3.12.
    "$venv/bin/pip" install --no-deps --no-build-isolation --ignore-requires-python -e "$src"
    "$venv/bin/python" -c "import torch, transformers, gr00t, onnx; print('groot:', 'torch', torch.__version__, '| transformers', transformers.__version__, '| onnx', onnx.__version__)"
  else
    "$venv/bin/python" -c "import torch, lerobot, onnx; print('$f:', 'torch', torch.__version__, '| lerobot', lerobot.__version__, '| onnx', onnx.__version__)"
  fi
done
