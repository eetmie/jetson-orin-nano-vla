# jetson-orin-nano-vla

**Tested on JetPack 7.2.1 (L4T R39.2.1).**

Recipes and measurements for running public base VLA models on an **8 GB Jetson
Orin Nano Super**. The comparison covers four public base-model profiles and the
trained EVO1 LIBERO profile:

| model | upstream checkpoint | split ONNX bundle |
|---|---|---|
| SmolVLA 450M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | `export/export.sh`; ORT also runs [`eetmie/smolvla-base-onnx`](https://huggingface.co/eetmie/smolvla-base-onnx) |
| EVO1 775M LIBERO | [`zuoxingdong/evo1_libero`](https://huggingface.co/zuoxingdong/evo1_libero) | `export/export.sh`; **trained** action head |
| X-VLA 0.9B | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | `export/export.sh`; ORT also runs [`eetmie/xvla-base-onnx`](https://huggingface.co/eetmie/xvla-base-onnx) |
| GR00T N1.7 3B | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) | `export/export.sh` |
| GR00T N1.6 3B | [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) | `export/export.sh` |

Every model runs on the **pure TensorRT** runtime (`bench trt-split`, no ONNX Runtime).

## Measured fit

Retained runs use pinned MAXN_SUPER clocks and deterministic synthetic observations.
They measure inference cost, not robot-task quality.

| model / runtime | views | p50 | p95 | rate | RAM in use |
|---|---:|---:|---:|---:|---:|
| SmolVLA PyTorch FP32 | 2 | 1167.93 ms | 1176.65 ms | 0.86 Hz | 4.13 GB |
| SmolVLA split, ORT FP16 | 2 | 189.89 ms | 190.93 ms | 5.25 Hz | 2.39 GB |
| **SmolVLA** split, pure TensorRT mixed FP16 | 2 | 185.28 ms | 186.86 ms | 5.39 Hz | 1.65 GB |
| EVO1 LIBERO split, ORT mixed FP16 | 2 | 414.67 ms | 424.72 ms | 2.41 Hz | 6.00 GB |
| **EVO1 LIBERO** split, pure TensorRT mixed FP16 | 2 | 413.93 ms | 414.75 ms | 2.42 Hz | 2.48 GB |
| X-VLA PyTorch FP32 | 3 | 2313.50 ms | 2320.89 ms | 0.43 Hz | 5.45 GB |
| X-VLA split, ORT FP16 | 3 | 391.55 ms | 407.33 ms | 2.55 Hz | 5.39 GB |
| **X-VLA** split, pure TensorRT mixed FP16 | 3 | 405.52 ms | 408.50 ms | 2.46 Hz | 2.77 GB |
| **GR00T N1.7 3B** split, pure TensorRT mixed FP16 | 3 (×2 frames) | 363.82 ms | 364.87 ms | 2.75 Hz | 5.88 GB |
| **GR00T N1.6 3B** split, pure TensorRT mixed FP16 | 3 | 349.75 ms | 366.42 ms | 2.84 Hz | 5.44 GB |

Less views make the model run faster. Single cam SmolVLA was sporting almost 7hz during robot usage!

A second Orin Nano Super, same JetPack and clock settings, ran X-VLA and EVO1 4-6 % slower (SmolVLA
matched), so YMMV. The pure TensorRT and GR00T rows come from that second board, the PyTorch and ORT
rows from the first. `RAM in use` is the whole system's RAM in use while inferring (`sys RAM` in the results).

The split bundles fit because the large policies are divided into independently built
TensorRT engines. A whole-policy TensorRT build exceeds the board's unified-memory
budget. Running those engines on TensorRT alone, without ONNX Runtime, holds each weight
once: about half the memory of the ORT runs, at about the same speed. That is what lets
GR00T N1.6 (3.3 B, ~2.3 B deployed) fit at all; it held 351.63 ms p50 / 353.05 ms p95
over a 5-minute sustained run. GR00T N1.7 (3.1 B, ~2.5 B deployed, Cosmos-Reason2
backbone) held 365.70 ms p50 / 367.38 ms p95. Every pure TensorRT model ran a 5-minute
sustained run within 0.6 % of its short-run p50. Full memory, power,
CPU, thermal, validity, and per-graph measurements are in
[the generated results](docs/RESULTS.md).

## Parity gate

Speed is only worth measuring if the actions survive the conversion. On this board that
is a live question rather than a formality: compute 8.7 makes FP16 the only fast reduced
precision available. Every backend is handed the *same*
seeded observations and the *same* injected noise (`bench/obs.py`), so the action chunks
line up element by element rather than only in distribution.

The measured values are in [the results](docs/RESULTS.md#parity). The short version: the
converted models reproduce their reference actions to **cosine 0.9993 or better, and
within 0.49 % of the action range on the executed action**.

`bench parity` gates the whole chunk, not only the executed action: max difference ≤ 1 %
of range. X-VLA passes. SmolVLA's 50-step chunk stays at 0.23 % (p95) and 0.62 % (p99), but
its single worst element reaches 2.05 %, so the command below reports FAIL for it.
Every pure TensorRT run is checked against the stock PyTorch FP32 policy by a fixture in
its bundle, every time it loads; the worst element over the full chunk is 0.14 % (SmolVLA),
0.06 % (EVO1 LIBERO), 0.05 % (X-VLA), 0.27 % (GR00T N1.6) and 0.09 % (GR00T N1.7) of range.

```bash
python -m bench parity results/smolvla-base.torch.json results/smolvla-base.ort.json \
    --reference smolvla-base.torch
```

It exits nonzero on a miss, and refuses any pair whose observations or injected noise
differ rather than reporting a cosine against a sequence the reference never saw. EVO1 is
the one model with no deployable PyTorch reference at all, so it is checked against a
native fixture carried inside its bundle, which fails closed during load.

## Run SmolVLA, X-VLA or EVO1 LIBERO

Export on the machine you fine-tune on, copy the bundle over whole, and run it on the
TensorRT runtime alone:

```bash
# on the export machine
export/setup.sh
export/export.sh lerobot/smolvla_base ~/bundles/smolvla-base-split --views 2
export/export.sh lerobot/xvla-base ~/bundles/xvla-base-split
export/export.sh zuoxingdong/evo1_libero ~/bundles/evo1-libero-split \
    --task "pick up the black bowl and place it on the plate"

# on the Orin
scripts/00_host_prep.sh
scripts/11_env_ort.sh
MODEL=smolvla-base BUNDLE=~/bundles/smolvla-base-split scripts/run_all.sh
MODEL=xvla-base    BUNDLE=~/bundles/xvla-base-split    scripts/run_all.sh
MODEL=evo1-libero  BUNDLE=~/bundles/evo1-libero-split  scripts/run_all.sh
```

The first run builds the engines one at a time: about 2 minutes for SmolVLA, 4 for
X-VLA and 3 for EVO1. Each bundle carries the stock PyTorch FP32 policy's output for
seeded inputs, and loading fails closed if the engines miss it. `run_all.sh` also runs
the PyTorch reference when its venv exists (`scripts/10_env_torch.sh` for SmolVLA,
`scripts/13_env_torch_xvla.sh` for X-VLA) and the checkpoint has been fetched with
`python -m bench fetch --model <model> --what torch`. `evo1-libero` is trained
([`zuoxingdong/evo1_libero`](https://huggingface.co/zuoxingdong/evo1_libero)), and its
actions mean something for LIBERO's embodiment and nothing else.

The SmolVLA and X-VLA ORT rows ran the Hugging Face bundles (`scripts/fetch_models.sh
smolvla-base`, then `bench ort-split`); EVO1's ran an earlier export of the same graphs.
X-VLA's ORT engine build is very memory-limited: **4 GB swap and a headless board are a
must**, and if it fails, reboot and build on the fresh board.
Details in [host setup](docs/01-host-setup.md#swap--4-gb-and-build-on-a-freshly-booted-headless-board).

## Run GR00T N1.6

GR00T is exported on the machine you fine-tune on (`export/README.md`), copied over
whole, and run on the TensorRT runtime alone: through ORT its weights would be held
twice and could not fit.

```bash
# on the export machine
export/setup.sh groot
export/export.sh nvidia/GR00T-N1.6-3B ~/bundles/groot-n16-base-split

# on the Orin
scripts/00_host_prep.sh
scripts/11_env_ort.sh
MODEL=groot-n16-base BUNDLE=~/bundles/groot-n16-base-split scripts/run_all.sh
```

The first run builds 26 engines, one at a time (about 7 minutes); every build kept at
least 3.9 GB free. The bundle carries stock-PyTorch FP32 outputs, and loading fails
closed if the engines miss them. The weights, and so the bundle, are under the NVIDIA
One-Way Noncommercial License.

## Run GR00T N1.7

Same route, own exporter venv. N1.7 reads its tokenizer and image processor from
[`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B), a gated
repository: accept its terms on Hugging Face before exporting.

```bash
# on the export machine
export/setup.sh groot17
export/export.sh nvidia/GR00T-N1.7-3B ~/bundles/groot-n17-base-split

# on the Orin
MODEL=groot-n17-base BUNDLE=~/bundles/groot-n17-base-split scripts/run_all.sh
```

26 engines again, about 6 minutes. N1.7 sees every camera twice, now and 30 frames
earlier; the runtime encodes only the new frames and reuses the earlier encode of the
history frame (an image's vision tokens depend on that image alone). The checkpoint's
license file is the NVIDIA License with a non-commercial use limitation.

## Export your own checkpoint

`export/` turns a LeRobot model checkpoint, base or fine-tuned, into a split bundle.
Run it on the machine you fine-tune on, not on the Jetson:

```
fetch model -> (fine-tune) -> export/export.sh -> copy bundle -> benchmark on the Orin
```

```bash
export/setup.sh
export/export.sh path/to/pretrained_model ~/bundles/my-policy
```

See [export/README.md](export/README.md).

## Documentation

- [Host setup](docs/01-host-setup.md)
- [Python environments](docs/02-environments.md)
- [Model and runtime contracts](docs/03-backends.md)
- [Metric definitions](docs/04-metrics.md)
- [Benchmark runbook](docs/05-runbook.md)
- [Measured results](docs/RESULTS.md)

## Scope

This repository runs and compares four public base checkpoints and the trained
EVO1 LIBERO checkpoint. `export/` lets you
measure yours. It does not contain training, fine-tuning, robot
control or camera capture. TensorRT engines are built on
the Jetson and are never copied between machines; the ONNX bundles are the portable
artifacts.

Repository code is MIT. Model weights and derived exports retain their respective
upstream licenses; consult each model card before redistribution.
