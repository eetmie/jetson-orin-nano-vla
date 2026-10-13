#!/usr/bin/env bash
# Optional: put the Jetson in the state the published numbers use.
#   MAXN_SUPER power mode, clocks pinned at its maximum, and optionally: pin them at
#   every boot, and boot without the desktop.
set -euo pipefail

mode=$(sed -n 's/.*POWER_MODEL ID=\([0-9]*\) NAME=MAXN_SUPER.*/\1/p' /etc/nvpmodel.conf)
[[ -n "$mode" ]] || { echo "no MAXN_SUPER power mode in /etc/nvpmodel.conf" >&2; exit 1; }
if ! nvpmodel -q | grep -q MAXN_SUPER; then
    # nvpmodel asks for a reboot itself when the switch needs one.
    sudo nvpmodel -m "$mode"
fi
sudo jetson_clocks
sudo jetson_clocks --show | grep -E "Power Mode|GPU MinFreq|Active GPU TPCs|EMC MinFreq"

if [[ -t 0 ]] && ! systemctl is-enabled --quiet jetson-clocks 2>/dev/null; then
    read -rp "Pin the clocks at every boot (installs a small systemd service)? [y/N] " answer
    if [[ "$answer" == [yY]* ]]; then
        sudo tee /etc/systemd/system/jetson-clocks.service >/dev/null <<'UNIT'
[Unit]
Description=Pin Jetson clocks at the power mode's maximum
After=nvpmodel.service nvpower.service

[Service]
Type=oneshot
# Late power-management udev events can reset the CPU floor; let them finish,
# and wait for the GPU to come up, before pinning.
ExecStartPre=/usr/bin/udevadm settle --timeout=30
ExecStartPre=/usr/bin/timeout 30 /bin/sh -c 'until [ -d /sys/class/devfreq/17000000.gpu ]; do sleep 1; done'
ExecStart=/usr/bin/jetson_clocks

[Install]
WantedBy=multi-user.target
UNIT
        sudo systemctl daemon-reload
        sudo systemctl enable jetson-clocks.service
        echo "Pinned at every boot. Undo: sudo systemctl disable jetson-clocks"
    else
        echo "Clocks stay pinned until the next reboot; run this again after one."
    fi
fi

if [[ -t 0 && "$(systemctl get-default)" == graphical.target ]]; then
    read -rp "Disable the desktop at boot? Frees about 0.1 GB of RAM (recommended) [y/N] " answer
    if [[ "$answer" == [yY]* ]]; then
        sudo systemctl set-default multi-user.target
        echo "Headless from the next reboot. Undo: sudo systemctl set-default graphical.target"
    fi
fi
