#!/usr/bin/env bash
# MAXN_SUPER + pinned clocks, with the check the old unit was missing.
#
# The previous jetson-perf.service ran `jetson_clocks` directly. That binary exits
# non-zero when the GPU has not finished initialising, so on a board whose nvgpu driver
# failed to bootstrap the unit failed at every boot and stayed failed -- while
# `nvpmodel -q` still cheerfully reported MAXN_SUPER. The board therefore LOOKED
# configured while every clock floated, and a whole set of benchmarks was taken that
# way before anyone noticed. Worse, the failure message is
# "Error! GPU frequency scaling not supported!", which reads like a platform
# limitation rather than a dead GPU.
#
# So: check the GPU is actually up, say plainly which case we are in, and verify the
# pin took rather than trusting the exit code.
set -uo pipefail

GPU_DEVFREQ=/sys/class/devfreq/17000000.gpu

nvpmodel -m 2 >/dev/null 2>&1 || true          # non-zero when already set; harmless
echo "power mode: $(nvpmodel -q 2>/dev/null | head -1)"

if [ ! -d "$GPU_DEVFREQ" ]; then
    echo "!! $GPU_DEVFREQ missing -- the GPU did not initialise this boot." >&2
    echo "   Clocks are NOT pinned and CUDA will be unavailable. Check:" >&2
    echo "     dmesg | grep -i 'ACR bootstrap\|acr_falcon2_sysmem_desc'" >&2
    echo "   An invalid WPR carveout needs a POWER CYCLE; a warm reboot may not clear" >&2
    echo "   it. See docs/01-host-setup.md." >&2
    exit 1
fi

jetson_clocks || { echo "!! jetson_clocks failed with the GPU present" >&2; exit 1; }

# Verify rather than trust: min == max is what "pinned" actually means.
cpu_min=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)
cpu_max=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
gpu_min=$(cat "$GPU_DEVFREQ/min_freq"); gpu_max=$(cat "$GPU_DEVFREQ/max_freq")
echo "cpu0 $((cpu_min/1000))-$((cpu_max/1000)) MHz   gpu $((gpu_min/1000000))-$((gpu_max/1000000)) MHz"
[ "$cpu_min" = "$cpu_max" ] && [ "$gpu_min" = "$gpu_max" ] \
    || { echo "!! clocks did not pin (min != max)" >&2; exit 1; }
echo "clocks pinned."
