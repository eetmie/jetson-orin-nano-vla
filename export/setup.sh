#!/usr/bin/env bash
# Create one exporter environment: export/setup.sh smolvla|xvla|groot|groot17
# ./export.sh calls this on first use. lerobot 0.5.1 (SmolVLA), 0.6.1 (X-VLA, EVO1,
# pi0.5 base), and GR00T N1.6's and N1.7's transformers cannot share one environment.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FAMS=("$@"); [[ ${#FAMS[@]} -gt 0 ]] || { echo "usage: export/setup.sh smolvla|xvla|groot|groot17"; exit 2; }
# uv's installer puts it in ~/.local/bin, which a shell opened before the install lacks.
command -v uv >/dev/null || PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "uv is missing: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
declare -A GROOT_REF=(
  [groot]=5dc80c4afd726b34faad1d8f7e007a13b34e4c88     # NVIDIA/Isaac-GR00T n1.6.1-release
  [groot17]=23ace64f17aa5015259b8609d371eb61a357c776   # NVIDIA/Isaac-GR00T n1.7-release
)
declare -A GROOT_SRC=([groot]=.isaac-gr00t [groot17]=.isaac-gr00t-n17)
for f in "${FAMS[@]}"; do
  venv="$HERE/.venv-$f"
  rm -rf "$venv"
  uv venv --quiet --python 3.12 "$venv"
  # Exact dependency closure, including indirect imports needed by the stock policy.
  uv pip install --quiet --python "$venv/bin/python" -r "$HERE/requirements-$f.txt"
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
    # On the path rather than installed: NVIDIA's package pins flash-attn/deepspeed for
    # training, which the export needs none of, and N1.7 declares Python 3.10 only.
    echo "$src" > "$("$venv/bin/python" -c "import sysconfig; print(sysconfig.get_path('purelib'))")/isaac-gr00t.pth"
    "$venv/bin/python" -c "import torch, transformers, gr00t, onnx; print('groot:', 'torch', torch.__version__, '| transformers', transformers.__version__, '| onnx', onnx.__version__)"
  else
    "$venv/bin/python" -c "import torch, lerobot, onnx; print('$f:', 'torch', torch.__version__, '| lerobot', lerobot.__version__, '| onnx', onnx.__version__)"
  fi
done
