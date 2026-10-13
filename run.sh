#!/usr/bin/env bash
# Jetson: build, verify and benchmark an exported bundle.   ./run.sh <bundle> [--help]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
# uv's installer puts it in ~/.local/bin, which a shell opened before the install lacks.
command -v uv >/dev/null || PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "uv is missing: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
if [[ -d "$ROOT/.venv/lib/python3.12/site-packages/triton" ]]; then   # installed
    uv sync --project "$ROOT" --frozen --quiet
else
    echo "Installing the Python environment (first run only, ~1 GB download)..."
    uv sync --project "$ROOT" --frozen
fi
PY="$ROOT/.venv/bin/python"

# TensorRT is not on PyPI for JetPack; use the system package's bindings.
if ! "$PY" -c "import tensorrt" 2>/dev/null; then
    trt=$(/usr/bin/python3 -c "import os, tensorrt; print(os.path.dirname(tensorrt.__file__))" 2>/dev/null) \
        || { echo "TensorRT's Python bindings are missing: sudo apt install python3-libnvinfer" >&2; exit 1; }
    ln -s "$trt" "$("$PY" -c "import sysconfig; print(sysconfig.get_path('purelib'))")/tensorrt"
fi

PYTHONPATH="$ROOT" PYTHONNOUSERSITE=1 exec "$PY" -m bench.run "$@"
