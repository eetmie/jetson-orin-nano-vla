#!/usr/bin/env bash
# Native model benchmark runtime; leaves existing ORT/PyTorch environments alone.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/native_env.sh runtime "${1:-.venv-trt}"
