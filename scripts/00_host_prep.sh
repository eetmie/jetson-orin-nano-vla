#!/usr/bin/env bash
# Put the Orin Nano into the state every number in this repo assumes.
#
#   MAXN_SUPER + pinned clocks   -> no DVFS ramp, no power cap, repeatable latency
#   16 GB swap                   -> the first TensorRT engine build fits in 8 GB
#   engine cache off /tmp        -> /tmp clears at boot; a rebuild is ~5 minutes
#
# Run once per boot (or install the systemd unit, below). Re-run `--verify` any time
# you doubt the board state — a benchmark taken at 15 W against one taken at MAXN is
# not a comparison.
set -euo pipefail

CACHE_DIR="${HOME}/.cache/jetson-orin-nano-vla/trt"

verify() {
    echo "== power mode ==";        sudo nvpmodel -q || true
    echo; echo "== clocks ==";      sudo jetson_clocks --show 2>/dev/null | head -6 || true
    echo; echo "== memory ==";      free -h
    echo; echo "== swap ==";        swapon --show || echo "(no swap!)"
    echo; echo "== thermal ==";     cat /sys/devices/virtual/thermal/thermal_zone*/type 2>/dev/null | paste -sd' '
    echo; echo "== L4T ==";         cat /etc/nv_tegra_release 2>/dev/null || true
    echo; echo "== tegrastats ==";  command -v tegrastats >/dev/null && echo "present" || echo "MISSING"
    echo; echo "== engine cache =="; echo "${CACHE_DIR}"; ls -la "${CACHE_DIR}" 2>/dev/null | head -5 || echo "(empty)"
}

if [[ "${1:-}" == "--verify" ]]; then verify; exit 0; fi

echo ">> nvpmodel -m 2 (MAXN_SUPER)  [0=15W 1=25W 2=MAXN_SUPER on JetPack 7.2]"
sudo nvpmodel -m "${1:-2}"

echo ">> jetson_clocks (pin every clock; removes the DVFS ramp from the first samples)"
sudo jetson_clocks

# 8 GB is UNIFIED — the CPU and the GPU share it, and a TensorRT build peaks well
# above what is free. Swap is what keeps the build from being an OOM kill.
if ! swapon --show | grep -q .; then
    echo "!! no swap configured. The first TRT engine build will likely OOM."
    echo "   See spark-projects/orin-nano/system/setup-swap.sh (16 GB on the NVMe)."
fi

mkdir -p "${CACHE_DIR}"

echo ">> stopping the desktop is optional (idle GNOME is only ~110 MB here) but"
echo "   a browser or an editor is not: close them before a build."
echo
verify
