#!/usr/bin/env bash
# Shared implementation for the native runtime/compiler setup scripts.
# Called after the wrapper has changed to the repository root.
set -euo pipefail

role="$1"
native_venv="$2"
manager="${VLA_ENV_MANAGER:-pip}"
system_python="${VLA_SYSTEM_PYTHON:-/usr/bin/python3}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH

case "$role" in
    runtime) requirements=requirements/trt.txt; lock=requirements/trt-uv.lock.txt ;;
    build) requirements=requirements/triton-build.txt; lock=requirements/triton-build-uv.lock.txt ;;
    *) echo "unknown native environment role: $role" >&2; exit 2 ;;
esac

case "$manager" in
    pip)
        "$system_python" -m bench.tools.trt_env "$native_venv"
        "$native_venv/bin/python" -m pip install pip==24.0
        "$native_venv/bin/python" -m pip install -r "$requirements"
        "$native_venv/bin/python" -m pip check
        ;;
    uv)
        uv_bin="${UV_BIN:-uv}"
        expected_uv="$(<.uv-version)"
        if ! command -v "$uv_bin" >/dev/null; then
            echo "uv $expected_uv is required; see docs/02-environments.md" >&2
            exit 2
        fi
        actual_uv="$("$uv_bin" --version)"
        if [[ "$actual_uv" != "uv $expected_uv" && "$actual_uv" != "uv $expected_uv "* ]]; then
            echo "expected uv $expected_uv; found $actual_uv" >&2
            exit 2
        fi
        [[ -f "$lock" ]] || { echo "missing dependency lock: $lock" >&2; exit 2; }
        # Use JetPack's interpreter, not an automatically downloaded Python.
        "$system_python" -m bench.tools.trt_env "$native_venv" --without-pip
        "$uv_bin" --no-config pip sync --python "$native_venv/bin/python" \
            --require-hashes --no-build "$lock"
        "$uv_bin" --no-config pip check --python "$native_venv/bin/python"
        ;;
    *) echo "VLA_ENV_MANAGER must be pip or uv; found $manager" >&2; exit 2 ;;
esac

"$native_venv/bin/python" -m bench.tools.doctor --role "$role"
