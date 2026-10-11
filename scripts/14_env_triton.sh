#!/usr/bin/env bash
# AOT builder, separate from deployed runtime and PyTorch reference environments.
set -euo pipefail
cd "$(dirname "$0")/.."
triton_venv="${1:-.venv-triton}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
python3 -m bench.tools.trt_env "$triton_venv"
"$triton_venv/bin/python" -m pip install pip==24.0
"$triton_venv/bin/python" -m pip install -r requirements/triton-build.txt
"$triton_venv/bin/python" -m pip check
"$triton_venv/bin/python" -m bench.tools.doctor --role build
