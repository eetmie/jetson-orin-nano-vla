#!/usr/bin/env bash
# PC side: export a checkpoint into a bundle for the Jetson.
#
#   ./export.sh <model> [checkpoint] [out-dir] [options]
#
#   model       smolvla | xvla | evo1 | groot-n16 | groot-n17 | pi05-base | pi05-libero
#   checkpoint  a Hugging Face id or a local LeRobot pretrained_model dir
#               (default: the public base checkpoint)
#   out-dir     default: bundles/<model>
#   options     --task "..." (repeatable), --views N, --fps N, --revision SHA; see export/README.md
#
# The first export of each model family creates its pinned environment in export/.venv-*.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
usage() { sed -n '4,12p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
[[ $# -ge 1 && "$1" != -* ]] || usage
model=$1; shift
case "$model" in
  smolvla)     env=smolvla; ckpt=lerobot/smolvla_base ;;
  xvla)        env=xvla;    ckpt=lerobot/xvla-base ;;
  evo1)        env=xvla;    ckpt=zuoxingdong/evo1_libero ;;
  pi05-base)   env=xvla;    ckpt=lerobot/pi05_base ;;
  groot-n16)   env=groot;   ckpt=nvidia/GR00T-N1.6-3B ;;
  groot-n17)   env=groot17; ckpt=nvidia/GR00T-N1.7-3B ;;
  pi05-libero) env=docker ;;
  *) usage ;;
esac
custom=""
if [[ $# -gt 0 && "$1" != -* ]]; then custom=$1; ckpt=$1; shift; fi
out="$ROOT/bundles/$model"
if [[ $# -gt 0 && "$1" != -* ]]; then out=$1; shift; fi
flags=()
if [[ "$model" == smolvla ]]; then
  # The AOT runtime is built for two camera slots and hoisted cross-attention K/V.
  flags=(--hoist-cross-kv)
  [[ " $* " == *" --views "* ]] || flags+=(--views 2)
fi

if [[ "$env" == docker ]]; then
  # openpi's pi0.5 LIBERO policy: converted and exported in its pinned container.
  [[ $# == 0 ]] || { echo "pi05-libero takes no options" >&2; exit 2; }
  exec "$ROOT/export/pi05/export.sh" "$out" ${custom:+"$custom"}
fi
[[ -x "$ROOT/export/.venv-$env/bin/python" ]] || "$ROOT/export/setup.sh" "$env"
exec "$ROOT/export/export.sh" "$ckpt" "$out" "${flags[@]}" "$@"
