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

## Swap and build memory

The 8 GB is unified memory: model weights, TensorRT build scratch, desktop applications,
and the runtime all compete for the same pool.

SmolVLA's split export was cold-built successfully with the stock 2 GB swapfile; swap
use peaked around 437 MB. X-VLA's FP32 bundle did not complete its first engine, while
the smaller FP16 bundle built successfully. Use the FP16 X-VLA bundle, keep swap
enabled, and close memory-heavy applications before building. Larger swap can provide
failure headroom, but it does not make an oversized engine a supported deployment path.

`scripts/setup-swap.sh` creates a single persistent swapfile (default 16 GB) that survives
reboot. X-VLA's twelve-engine build needs it; SmolVLA's three-engine split does not.

```bash
sudo ./scripts/setup-swap.sh
```

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
