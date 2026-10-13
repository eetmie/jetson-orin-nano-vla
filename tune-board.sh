#!/usr/bin/env bash
# Optional: put the Jetson in the state the published numbers use.
#   MAXN_SUPER power mode, clocks pinned at its maximum, and optionally no desktop.
# Pinned clocks reset at reboot; run this again after each one.
set -euo pipefail

mode=$(sed -n 's/.*POWER_MODEL ID=\([0-9]*\) NAME=MAXN_SUPER.*/\1/p' /etc/nvpmodel.conf)
[[ -n "$mode" ]] || { echo "no MAXN_SUPER power mode in /etc/nvpmodel.conf" >&2; exit 1; }
if ! nvpmodel -q | grep -q MAXN_SUPER; then
    # nvpmodel asks for a reboot itself when the switch needs one.
    sudo nvpmodel -m "$mode"
fi
sudo jetson_clocks
sudo jetson_clocks --show | grep -E "Power Mode|GPU MinFreq|Active GPU TPCs|EMC MinFreq"

if [[ -t 0 && "$(systemctl get-default)" == graphical.target ]]; then
    read -rp "Disable the desktop at boot? Frees about 0.4 GB of RAM (recommended) [y/N] " answer
    if [[ "$answer" == [yY]* ]]; then
        sudo systemctl set-default multi-user.target
        echo "Headless from the next reboot. Undo: sudo systemctl set-default graphical.target"
    fi
fi
