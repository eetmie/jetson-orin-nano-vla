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
set -euo pipefail

GPU_DEVFREQ=/sys/class/devfreq/17000000.gpu

verify_only=false
case "${1:-}" in
    --verify) verify_only=true ;;
    "") ;;
    *) echo "usage: jetson-perf.sh [--verify]" >&2; exit 2 ;;
esac

if ! "$verify_only"; then
    # Some releases return nonzero when the mode is already selected. Verify
    # the actual mode below rather than treating every failure as harmless.
    nvpmodel -m 2 || echo "nvpmodel returned nonzero; checking the selected mode" >&2
fi
power_mode=$(nvpmodel -q)
echo "$power_mode"
[[ "$power_mode" == *"NV Power Mode: MAXN_SUPER"* ]] \
    || { echo "!! MAXN_SUPER was not selected" >&2; exit 1; }

# Wait for driver startup at boot instead of ordering this service after the
# target that also wants to start it. Verification never waits or changes clocks.
if ! "$verify_only"; then
    for attempt in {1..30}; do
        [[ -d "$GPU_DEVFREQ" ]] && break
        sleep 1
    done
fi

if [ ! -d "$GPU_DEVFREQ" ]; then
    echo "!! $GPU_DEVFREQ missing -- the GPU did not initialise this boot." >&2
    echo "   Clocks are NOT pinned and CUDA will be unavailable. Check:" >&2
    echo "     dmesg | grep -i 'ACR bootstrap\|acr_falcon2_sysmem_desc'" >&2
    echo "   An invalid WPR carveout may need a POWER CYCLE if a warm reboot does" >&2
    echo "   not clear it. See docs/01-host-setup.md." >&2
    exit 1
fi

# The saved mode name can be correct even when boot-time GPU power gating
# failed. These masks are required by this Orin Nano Super's MAXN_SUPER profile.
for mask in tpc:240 fbp:2; do
    name=${mask%%:*}; expected=${mask##*:}
    actual=$(cat "/sys/devices/platform/gpu.0/${name}_pg_mask")
    [[ "$actual" == "$expected" ]] \
        || { echo "!! ${name}_pg_mask=$actual; expected $expected. Resolve boot power ordering and reboot." >&2; exit 1; }
done

if ! "$verify_only"; then
    jetson_clocks || { echo "!! jetson_clocks failed with the GPU present" >&2; exit 1; }
fi

# Verify rather than trust: min == max is what "pinned" actually means.
[[ "$(cat /sys/devices/system/cpu/online)" == "0-5" ]] \
    || { echo "!! expected all six Orin Nano CPU cores online" >&2; exit 1; }
for cpu in /sys/devices/system/cpu/cpu[0-5]; do
    cpu_min=$(cat "$cpu/cpufreq/scaling_min_freq")
    cpu_max=$(cat "$cpu/cpufreq/scaling_max_freq")
    echo "${cpu##*/} $((cpu_min/1000))-$((cpu_max/1000)) MHz"
    [[ "$cpu_min" == 1728000 && "$cpu_max" == "$cpu_min" ]] \
        || { echo "!! ${cpu##*/} is not pinned at the MAXN_SUPER frequency" >&2; exit 1; }
done
gpu_min=$(cat "$GPU_DEVFREQ/min_freq"); gpu_max=$(cat "$GPU_DEVFREQ/max_freq")
echo "gpu $((gpu_min/1000000))-$((gpu_max/1000000)) MHz"
[[ "$gpu_min" == 1020000000 && "$gpu_max" == "$gpu_min" ]] \
    || { echo "!! GPU is not pinned at the MAXN_SUPER frequency" >&2; exit 1; }
clock_state=$(jetson_clocks --show)
echo "$clock_state"
emc_line=$(printf '%s\n' "$clock_state" | grep '^EMC ')
[[ "$emc_line" == *"MinFreq=3199000000 MaxFreq=3199000000 CurrentFreq=3199000000"* ]] \
    || { echo "!! EMC is not pinned at the MAXN_SUPER frequency" >&2; exit 1; }
echo "clocks pinned."
