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

## Swap: 16 GB, on the NVMe

The 8 GB is **unified** — CPU and GPU share it. TensorRT's first engine build is the
memory peak of this whole exercise, and it peaks well above what is free. Without
swap it is an OOM kill; with 16 GB it completes.

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
