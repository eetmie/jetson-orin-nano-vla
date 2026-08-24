#!/usr/bin/env bash
# venv for the split-ONNX / TensorRT-EP backend.
#
# --system-site-packages is REQUIRED: `tensorrt` ships with JetPack and is not
# pip-installable, and pyrealsense2 (if you ever point this at a live camera) comes
# from the librealsense RSUSB build. A clean venv cannot see either.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${1:-.venv-ort}"
INDEX="https://pypi.jetson-ai-lab.io/sbsa/cu130"

python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -U pip wheel
"$VENV/bin/pip" install -r requirements/ort-split.txt --extra-index-url "$INDEX"

"$VENV/bin/python" - <<'PY'
import tensorrt, onnxruntime as ort
print("tensorrt", tensorrt.__version__)
print("onnxruntime", ort.__version__)
eps = ort.get_available_providers()
print("providers", eps)
assert "TensorrtExecutionProvider" in eps, (
    "the TensorRT EP did not register. Without it this backend falls back to CUDA or "
    "CPU and measures something else entirely.")
PY
echo "OK -> $VENV"
