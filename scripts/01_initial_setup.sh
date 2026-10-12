#!/usr/bin/env bash
# Provision the tested JetPack image; inspect by default, mutate only with --apply.
# Run apply/verify with sudo. SSH and Tailscale remain running.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=--inspect
disable_docker=false
reboot_if_needed=false
board_hostname=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --inspect|--apply|--verify) mode="$1"; shift ;;
        --disable-docker) disable_docker=true; shift ;;
        --reboot-if-needed) reboot_if_needed=true; shift ;;
        --hostname) board_hostname="${2:?--hostname requires a name}"; shift 2 ;;
        *) echo "usage: $0 [--inspect|--apply|--verify] [--disable-docker] [--reboot-if-needed] [--hostname NAME]" >&2; exit 2 ;;
    esac
done

inspect() {
    hostname
    cat /etc/nv_tegra_release
    systemctl get-default
    free -m
    swapon --show
    nvpmodel -q
    systemctl is-active ssh.service tailscaled.service nvfancontrol.service || true
    dpkg-query -W python3-venv python3-libnvinfer libnvinfer10 libnvonnxparsers10 cuda-cudart-13-2
}
if [[ "$mode" == --inspect ]]; then inspect; exit 0; fi
[[ "$EUID" == 0 ]] || { echo "$mode requires sudo" >&2; exit 1; }
grep -Eq '^# R39 .*REVISION: 2\.1,' /etc/nv_tegra_release \
    || { echo "this recipe targets L4T R39.2.1; inspect the platform before provisioning" >&2; exit 1; }
[[ "$(tr -d '\0' </proc/device-tree/model)" == *"Orin Nano"*"Super"* ]] \
    || { echo "this recipe targets the Orin Nano Super" >&2; exit 1; }

# Fail before changing the host if its GPU or native dependencies are unavailable.
/usr/bin/python3 - <<'PY'
import ctypes, importlib.util, sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit("JetPack Python 3.12 is required")
if importlib.util.find_spec("tensorrt") is None:
    raise SystemExit("install JetPack's TensorRT Python bindings first")
import tensorrt
if tensorrt.__version__ != "10.16.2.10":
    raise SystemExit(f"this recipe requires TensorRT 10.16.2.10; found {tensorrt.__version__}")
cuda = ctypes.CDLL("libcuda.so.1")
version = ctypes.c_int()
rc = cuda.cuDriverGetVersion(ctypes.byref(version))
if rc or version.value != 13020:
    raise SystemExit(f"this recipe requires CUDA driver 13.2; found {version.value} (status {rc})")
rc = cuda.cuInit(0)
if rc:
    raise SystemExit(f"CUDA initialization failed ({rc}); resolve GPU startup before provisioning")
PY

verify() {
    [[ "$(systemctl get-default)" == multi-user.target ]] \
        || { echo "headless boot target is not configured" >&2; return 1; }
    if systemctl is-active --quiet display-manager.service; then
        echo "display manager is still active" >&2; return 1
    fi
    for service in ssh.service tailscaled.service nvfancontrol.service; do
        systemctl is-active --quiet "$service" \
            || { echo "$service is not active" >&2; return 1; }
    done
    systemctl is-enabled --quiet jetson-perf.service \
        || { echo "jetson-perf.service is not enabled" >&2; return 1; }
    systemctl is-active --quiet jetson-perf.service \
        || { echo "jetson-perf.service failed or is inactive; inspect journalctl -b -u jetson-perf.service" >&2; return 1; }
    [[ "$(systemctl show nvpmodel.service -p Result --value)" == success ]] \
        || { echo "NVIDIA's boot power service failed; resolve startup ordering and reboot" >&2; return 1; }
    if "$disable_docker"; then
        for service in docker.socket docker.service containerd.service nvidia-cdi-refresh.path nvidia-cdi-refresh.service; do
            if systemctl cat "$service" >/dev/null 2>&1; then
                if systemctl is-enabled --quiet "$service" || systemctl is-active --quiet "$service"; then
                    echo "$service is still enabled/active" >&2; return 1
                fi
            fi
        done
    fi
    [[ -f /swapfile && ! -L /swapfile && "$(stat -c %s /swapfile)" == 4294967296 ]] \
        || { echo "expected a 4 GiB /swapfile" >&2; return 1; }
    [[ "$(stat -c %a /swapfile)" == 600 ]] \
        || { echo "/swapfile must have mode 600" >&2; return 1; }
    swapon --noheadings --raw --show=NAME | grep -Fxq /swapfile \
        || { echo "/swapfile is not active" >&2; return 1; }
    /usr/local/sbin/jetson-perf.sh --verify
    inspect
    echo "host ready: headless, MAXN_SUPER, pinned clocks, 4 GiB swap, SSH/Tailscale active"
}
if [[ "$mode" == --verify ]]; then verify; exit 0; fi

for service in ssh.service tailscaled.service; do
    systemctl is-active --quiet "$service" \
        || { echo "$service must be active before provisioning" >&2; exit 1; }
done
if "$disable_docker" && command -v docker >/dev/null; then
    # A fresh image normally has no containers. Refuse to stop unrelated workloads.
    if systemctl is-active --quiet docker.service; then
        [[ -z "$(docker ps -q)" ]] \
            || { echo "running Docker containers found; refusing to stop their services" >&2; exit 1; }
    fi
fi

if [[ -L /swapfile ]]; then echo "refusing a symlink at /swapfile" >&2; exit 1; fi
if [[ ! -f /swapfile || "$(stat -c %s /swapfile)" != 4294967296 ]]; then
    # Prepare the replacement before disabling the old swap. Check that bringing
    # its pages into RAM has headroom, and preserve fstab before editing it.
    /usr/bin/python3 - <<'PY'
from pathlib import Path
mem = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
if mem["MemAvailable"] < mem["SwapTotal"] - mem["SwapFree"] + 512 * 1024:
    raise SystemExit("not enough available RAM to resize swap safely; stop workloads first")
PY
    swap_tmp=$(mktemp /.vla-swap.XXXXXXXX)
    trap '[[ -z "${swap_tmp:-}" ]] || rm -f -- "$swap_tmp"' EXIT
    chmod 600 "$swap_tmp"
    fallocate -l 4G "$swap_tmp"
    mkswap "$swap_tmp"
    if swapon --noheadings --raw --show=NAME | grep -Fxq /swapfile; then swapoff /swapfile; fi
    mv -f -- "$swap_tmp" /swapfile
    swap_tmp=""
fi
chmod 600 /swapfile
if ! swapon --noheadings --raw --show=NAME | grep -Fxq /swapfile; then swapon /swapfile; fi
/usr/bin/python3 - <<'PY'
import os, shutil, tempfile
from pathlib import Path
p = Path("/etc/fstab")
backup = p.with_name("fstab.vla-before-setup")
if not backup.exists(): shutil.copy2(p, backup)
lines = p.read_text().splitlines()
lines = [line for line in lines if not (line.strip() and not line.lstrip().startswith("#") and line.split()[0] == "/swapfile")]
new = "\n".join(lines + ["/swapfile none swap sw 0 0"]) + "\n"
if new != p.read_text():
    fd, name = tempfile.mkstemp(prefix=".fstab-vla-", dir=p.parent)
    with os.fdopen(fd, "w") as f: f.write(new)
    os.chmod(name, p.stat().st_mode & 0o777)
    os.replace(name, p)
PY

if [[ -n "$board_hostname" ]]; then hostnamectl set-hostname "$board_hostname"; fi
systemctl enable ssh.service tailscaled.service nvfancontrol.service
systemctl set-default multi-user.target
systemctl stop display-manager.service
if "$disable_docker"; then
    # CDI refresh probes the GPU at boot even without Docker running. On R39.2
    # this can create its golden context before nvpmodel sets the TPC mask.
    for service in docker.socket docker.service containerd.service nvidia-cdi-refresh.path nvidia-cdi-refresh.service; do
        if systemctl cat "$service" >/dev/null 2>&1; then systemctl disable --now "$service"; fi
    done
fi
install -m 755 scripts/jetson-perf.sh /usr/local/sbin/jetson-perf.sh
install -m 644 scripts/jetson-perf.service /etc/systemd/system/jetson-perf.service
systemctl daemon-reload
systemctl enable jetson-perf.service
if [[ "$(nvpmodel -q)" != *"NV Power Mode: MAXN_SUPER"* \
   || "$(cat /sys/devices/platform/gpu.0/tpc_pg_mask)" != 240 \
   || "$(cat /sys/devices/platform/gpu.0/fbp_pg_mask)" != 2 \
   || "$(systemctl show nvpmodel.service -p Result --value)" != success ]]; then
    # TPC masks cannot change after a GPU context exists. The fresh desktop image
    # therefore needs a reboot for its first 25W -> MAXN_SUPER transition.
    if ! "$reboot_if_needed"; then
        echo "host configured; MAXN_SUPER transition requires an authorized reboot" >&2
        echo "rerun --apply --reboot-if-needed, then reconnect and run --verify" >&2
        exit 75
    fi
    echo "selecting MAXN_SUPER; the board will reboot and pin clocks at startup"
    nvpmodel -m 2 --force
    # Ensure a reboot/persistence check even if this release applied the mode
    # without rebooting inside nvpmodel.
    systemctl reboot
    exit 0
fi
systemctl restart jetson-perf.service
verify
echo "reboot, reconnect over SSH, then run: sudo scripts/01_initial_setup.sh --verify"
