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

## Swap: 16 GB for the monolith, stock 2 GB is enough for the split

The 8 GB is **unified** — CPU and GPU share it, and TensorRT's first engine build is
where the monolithic export ran out of room: without swap it was an OOM kill. 16 GB is
what the *monolithic* build attempts were done with.

**SmolVLA's split does not need it — measured 2026-08-25.** A cold `ort-split` run on a
fresh JetPack 7.2 board with the **stock 2 GB swapfile** built all three heavy engines
(vision, expert-prefill, expert-decode; 719 MB cached total, `fp16_sm87`) with no OOM,
no thrash and no swap spike — swap use peaked at ~437 MB and the board never dropped
below ~1.4 GB available.

**X-VLA's split is a different question — read this before assuming "the split is
fine".** Twelve engines and 875 M params, not three engines and 450 M. An `ort-split`
run against the **FP32** X-VLA bundle on the stock 2 GB swapfile reached 6055 MB RSS
with 169 MB free and 1154 MB of swap consumed, and had built **zero** engines — it was
thrashing on the first one and was killed rather than left to hit the OOM killer. The
subprocess-per-graph isolation was working correctly; it is not what saves you here.

What fixed it: the **FP16 bundle** (`tools/fp16_weights.py`, 3503 -> 1753 MB), run with
16 GB of swap available. Engines then built with **swap essentially untouched (1 MB
used, ~4.1 GB available)** — so on this evidence the FP16 bundle is doing the work and
the extra swap was headroom that never got called on. Both were changed at once, so
which one is strictly necessary is untested; if you want the answer, try FP16 on the
stock 2 GB. Note that FP16 is *not* expected to reduce the TRT build peak — TensorRT
imports weights as FP32 working copies whatever the file dtype — so the mechanism here
is probably the smaller ONNX parse rather than the engine build itself.

Grow swap when you are attempting the monolith, or anything X-VLA-sized; the stock 2 GB
is enough for SmolVLA's split.

Worth knowing why the peak is what it is: for the *monolithic* SmolVLA export the
build peak is a node-count-independent floor of roughly 6 GB, because TensorRT
imports all 450M weights as FP32 working copies at once. That is why the monolith
does not build on this board at any precision or step count, and why the deploy path
is per-component engines. See `docs/03-backends.md`.

## Engine cache off /tmp

`/tmp` clears at boot, and a cold TensorRT build is ~5 minutes. The harness defaults
to `~/.cache/jetson-orin-nano-vla/trt`, which survives reboots. Delete it deliberately
when you want to measure a cold build; do not let a reboot decide for you.

## Before a build, close things

The memory is shared, so a browser or an editor eats directly into what TensorRT can
use. Stopping the desktop entirely buys little here — idle GNOME measured only ~110 MB
on this board, not the ~800 MB the Jetson AI Lab notes assume — but an open
application is a different matter. Run one thing at a time.

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
