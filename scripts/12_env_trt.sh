#!/usr/bin/env bash
# Native model benchmark runtime; leaves existing ORT/PyTorch environments alone.
set -euo pipefail
cd "$(dirname "$0")/.."
trt_venv="${1:-.venv-trt}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
python3 -m bench.tools.trt_env "$trt_venv"
"$trt_venv/bin/python" -m pip install pip==24.0
"$trt_venv/bin/python" -m pip install -r requirements/trt.txt
"$trt_venv/bin/python" -m pip check
"$trt_venv/bin/python" -m bench.tools.doctor --role runtime
