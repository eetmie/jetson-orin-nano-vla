#!/usr/bin/env bash
# Build the pinned container, download/convert the pinned policy, export a new bundle.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
RESUME=0
if [[ "${1:-}" == --resume-download ]]; then RESUME=1; shift; fi
OUT=${1:?usage: export/pi05/export.sh [--resume-download] OUT [CONVERTED_CHECKPOINT]}
OUT=$(realpath -m "$OUT")
WORK="$OUT.work"
[[ ! -e "$OUT" ]] || { echo "OUT must be a new directory"; exit 1; }
if [[ "$RESUME" == 1 ]]; then
  [[ $# == 1 && -d "$WORK/jax-checkpoint" && ! -e "$WORK/checkpoint" && ! -e "$WORK/stages" ]] || { echo "--resume-download requires an interrupted remote download before conversion/export"; exit 1; }
else
  [[ ! -e "$WORK" ]] || { echo "OUT.work must be a new directory; interrupted downloads can use --resume-download"; exit 1; }
fi
mkdir -p "$WORK"
docker build --iidfile "$WORK/image-id" -f "$HERE/Dockerfile" "$ROOT"
IMAGE=$(cat "$WORK/image-id")
if [[ $# -ge 2 ]]; then
  CHECKPOINT=$(cd "$2" && pwd)
else
  DOWNLOAD_ARGS=()
  [[ "$RESUME" == 1 ]] && DOWNLOAD_ARGS=(--resume)
  python3 "$HERE/download.py" --out "$WORK/jax-checkpoint" "${DOWNLOAD_ARGS[@]}"
  docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e OPENPI_DATA_HOME=/work/cache -v "$WORK:/work" -v "$ROOT:/repo:ro" "$IMAGE" \
    python /repo/export/pi05/convert.py --config_name pi05_libero \
      --checkpoint_dir /work/jax-checkpoint --output_path /work/checkpoint
  cp -r "$WORK/jax-checkpoint/assets" "$WORK/checkpoint/assets"
  CHECKPOINT="$WORK/checkpoint"
fi
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e OPENPI_DATA_HOME=/work/cache -e HF_HOME=/work/hf-cache \
  -v "$ROOT:/repo:ro" -v "$CHECKPOINT:/checkpoint:ro" -v "$WORK:/work" \
  -v "$(dirname "$OUT"):$(dirname "$OUT")" "$IMAGE" \
  python /repo/export/pi05/export.py --checkpoint /checkpoint --out "$OUT" --work-dir /work/stages
