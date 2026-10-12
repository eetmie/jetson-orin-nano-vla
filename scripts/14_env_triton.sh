#!/usr/bin/env bash
# AOT builder, separate from deployed runtime and PyTorch reference environments.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/native_env.sh build "${1:-.venv-triton}"
