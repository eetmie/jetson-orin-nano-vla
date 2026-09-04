# 1. Host setup — Jetson Orin Nano Super 8 GB, JetPack 7.2

Every number in this repo is only meaningful against a known board state. JetPack 7.2
is L4T R39.2.0: Ubuntu 24.04, kernel 6.8, Python 3.12, CUDA 13.2, cuDNN 9.20,
TensorRT 10.16.2. It is the first 7.x release to cover the whole Orin family, and the
Orin Nano dev kit no longer ships an SD-card image — flash with the USB ISO installer
or SDK Manager.

```bash
scripts/00_host_prep.sh            # set it
scripts/00_host_prep.sh --verify   # check it, any time you doubt a number
```

## Power mode: MAXN_SUPER, clocks pinned

```
nvpmodel IDs on this board:  0 = 15 W   1 = 25 W   2 = MAXN_SUPER
```

`nvpmodel -m 2 && jetson_clocks`. Two separate things: the first lifts the power cap,
the second pins clocks so DVFS does not spend the first seconds of every run ramping.
Without both, early iterations are slower than late ones and the "thermal drift"
column measures the governor rather than the thermals.

A run taken at 15 W and a run taken at MAXN are not comparable. The power mode is
recorded into every result JSON (`env.nvpmodel`) precisely so this cannot be
argued about after the fact.

### Verify the pinning actually took — running the command is not evidence

`jetson_clocks` exits non-zero and pins nothing if the GPU never finished
initialising, and `nvpmodel -m 2` still reports MAXN_SUPER afterwards, so the board
looks configured while every clock floats. Check the values, not the commands:

```bash
sudo jetson_clocks --show | grep -E "^cpu0|GPU|EMC"     # want MinFreq == MaxFreq
cat /sys/class/devfreq/17000000.gpu/cur_freq            # missing file = dead GPU
```

If `/sys/class/devfreq/17000000.gpu/` does not exist, the GPU did not come up and
nothing below this line is measurable. The signature is in `dmesg`:

```
nvgpu: 17000000.gpu  invalid mem acr_falcon2_sysmem_desc
nvgpu: RISCV ucode patch wpr info failed
nvgpu: ACR bootstrap failed
nvgpu: Failed initialization for: g->ops.acr.acr_construct_execute
```

That is the GPU's secure firmware failing to bootstrap because the WPR carveout the
bootloader handed it is invalid. `torch.cuda.is_available()` is False,
`/dev/nvgpu/igpu0/` holds only `power` instead of a dozen nodes, and `libnvrm_gpu.so:
NvRmGpuLibOpen failed` prefixes every command. **A warm `reboot` may not clear it** —
the carveout is built by the bootloader, so power-cycle the board. Observed once on
JetPack 7.2 (L4T R39.2.1) with a clean capsule status and both slots normal, i.e. not
a failed update, and it did not recur after a cold boot.

A systemd unit that runs `jetson_clocks` at boot will fail on a board in that state
and stay failed, which is worth knowing before concluding the platform cannot pin
clocks: check `systemctl status` for the unit before believing it.

`scripts/jetson-perf.sh` is written for exactly this: it checks the GPU devfreq node,
names the ACR diagnosis when the node is missing, and verifies `min == max` rather than
trusting the exit code of `jetson_clocks`. Install it to run at every boot:

```bash
sudo install -m 755 scripts/jetson-perf.sh /usr/local/sbin/jetson-perf.sh
sudo cp scripts/jetson-perf.service /etc/systemd/system/jetson-perf.service
sudo systemctl enable --now jetson-perf.service
systemctl status jetson-perf.service      # believe this, not the absence of an error
```

A warm reboot may not clear an ACR failure — power-cycle the board.

## Swap — 4 GB, and build on a freshly booted headless board

The 8 GB is unified memory: model weights, TensorRT build scratch, the desktop if you are
running one, and the runtime all compete for one pool. The single heaviest thing in this
repo is cold-building X-VLA's twelve FP16 engines, and it decides the swap size.

**Raise the stock 2 GB swapfile to 4 GB. Do not bother going higher.**

```bash
sudo swapoff /swapfile && sudo rm /swapfile
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile     # /etc/fstab already points at it
```

JetPack ships 2 GB via `nvfb-swapfile.service` (`fallocate -l 2G`, first boot only, so it
does not recreate the file if you replace it).

### Why 4 GB — the same build, four times

| swap | board state | peak swap used | min RAM free | result |
|---:|---|---:|---:|---|
| 2 GB | fresh boot, headless | 825 MB | 158 MB | **failed** on the last engine |
| 4 GB | fresh boot, headless | 867 MB | 200 MB | succeeded |
| 16 GB | fresh boot | 1812 MB | 65 MB | succeeded |
| 16 GB | already loaded | 1467 MB | 114 MB | **failed** on the last engine |

Note what the failures are *not*: neither exhausted its swap. The 2 GB run failed with
1.2 GB of its swapfile still free. The allocations that fail are `NvMap` — GPU memory,
which is physically backed and cannot be paged out — and the kernel's willingness to grant
them tracks *total* swap, not swap in use. That is why 2 GB fails and 4 GB works while
both use under 900 MB, and why 16 GB buys nothing over 4 GB.

### Build headless, on a freshly booted board

Board state matters as much as swap: the 16 GB run failed when the board had been working
beforehand and succeeded after a reboot. Every cold build here ran the machine to within
~200 MB of its ceiling.

- run headless (`systemctl get-default` should say `multi-user.target`)
- reboot first, then build, before anything else is resident
- **if a build fails, reboot and try again** — that is the fix, not a bigger swapfile

X-VLA and EVO1 are the two that need this; SmolVLA's three-graph split is far lighter and
peaks around 437 MB of swap. X-VLA's FP32 bundle does not complete its first engine at
all — use the FP16 bundle.

### The build is what runs you out of memory, not the running model

Once the engines are cached, X-VLA loads to 6.4 GB and sits with ~2.5 GB free at idle;
building them peaks at 7.0-7.4 GB. So an OOM here means "build the cache on an idle
board", not "X-VLA does not fit". Keep the cache (see below) and it does not recur.

## Engine cache off /tmp

`/tmp` clears at boot, and a cold TensorRT build is ~5 minutes. The harness defaults
to `~/.cache/jetson-orin-nano-vla/trt`, which survives reboots. Delete it deliberately
when you want to measure a cold build; do not let a reboot decide for you.

## Before a build, close things

The memory is shared, so a browser or an editor eats directly into what TensorRT can
use. Stopping the desktop entirely buys little here — idle GNOME measured only ~110 MB
on this board, not the ~800 MB the Jetson AI Lab notes assume — but an open
application is a different matter. Run one thing at a time.

## Will a given graph build? — the probes

Before splitting a new model, measure rather than guess. On this board the TensorRT build
peak tracks the weight slice an engine carries:

```bash
python bench/tools/build_probe.py --blocks 4 8 12     # build-memory curve, no checkpoint needed
python bench/tools/memory_probe.py --split-dir ~/bundles/xvla-base-split
```

`build_probe.py` is what produced the sizing rule the X-VLA split follows; `memory_probe.py`
decomposes resident memory once a bundle exists.

## Measurement tooling

`tegrastats` ships with L4T and is the only source that reports RAM, per-core CPU, GPU
utilisation, junction temperature and the board power rails together. The harness
parses it directly rather than depending on `jetson-stats`/`jtop`, which lags new
JetPack releases. Check it is present:

```bash
python -m bench selftest --seconds 4
```

That prints the sampler it chose and the numbers it read. If it says `PsutilMonitor`
on the Jetson, `tegrastats` is not on PATH and the power/GPU columns will be empty.
