#!/usr/bin/env bash
# Host PC: inspect a checkpoint and export its portable benchmark bundle.
# ./export.sh <checkpoint directory | Hugging Face URL/id | built-in alias> [options]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
exec python3 "$ROOT/export/cli.py" "$@"
