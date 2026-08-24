#!/usr/bin/env bash
# venv for FastCrest Tether. Kept separate on purpose: it depends on both torch and
# onnxruntime and would otherwise fight the other two environments over the two
# wheels that are hardest to reinstall on this board.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${1:-.venv-tether}"

python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -U pip wheel
"$VENV/bin/pip" install -r requirements/tether.txt

"$VENV/bin/tether" --version || true
"$VENV/bin/tether" doctor || echo "!! tether doctor reported problems — read them before benchmarking"
echo "OK -> $VENV   (tether doctor output above is part of the result; keep it)"
