# jetson-orin-nano-vla

**Tested on JetPack 7.2.1 (L4T R39.2.1).**

Recipes and measurements for running VLA models on an **8 GB Jetson Orin Nano Super**:
four public base checkpoints and the trained EVO1 LIBERO checkpoint, all on a pure
TensorRT runtime (`bench trt-split`).

| model | upstream checkpoint |
|---|---|
| SmolVLA 450M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| EVO1 775M LIBERO | [`zuoxingdong/evo1_libero`](https://huggingface.co/zuoxingdong/evo1_libero) |
| X-VLA 0.9B | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) |
| GR00T N1.7 3B | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| GR00T N1.6 3B | [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) |

## Measured fit

Pinned MAXN_SUPER clocks, deterministic synthetic observations, default denoising steps.
These measure inference cost, not robot-task quality.

| model / runtime | views | p50 | p95 | rate | RAM in use |
|---|---:|---:|---:|---:|---:|
| SmolVLA PyTorch FP32 | 2 | 1167.93 ms | 1176.65 ms | 0.86 Hz | 4.13 GB |
| SmolVLA split, ORT FP16 | 2 | 189.89 ms | 190.93 ms | 5.25 Hz | 2.39 GB |
| **SmolVLA** split, pure TensorRT mixed FP16 | 2 | 148.94 ms | 149.93 ms | 6.71 Hz | 1.80 GB |
| EVO1 LIBERO split, ORT mixed FP16 | 2 | 414.67 ms | 424.72 ms | 2.41 Hz | 6.00 GB |
| **EVO1 LIBERO** split, pure TensorRT mixed FP16 | 2 | 360.33 ms | 360.77 ms | 2.78 Hz | 2.56 GB |
| X-VLA PyTorch FP32 | 3 | 2313.50 ms | 2320.89 ms | 0.43 Hz | 5.45 GB |
| X-VLA split, ORT FP16 | 3 | 391.55 ms | 407.33 ms | 2.55 Hz | 5.39 GB |
| **X-VLA** split, pure TensorRT mixed FP16 | 3 | 383.68 ms | 385.10 ms | 2.61 Hz | 2.78 GB |
| **GR00T N1.7 3B** split, pure TensorRT mixed FP16 | 3 (×2 frames) | 288.11 ms | 289.09 ms | 3.47 Hz | 5.97 GB |
| **GR00T N1.6 3B** split, pure TensorRT mixed FP16 | 3 | 288.09 ms | 289.10 ms | 3.47 Hz | 5.49 GB |

The pure TensorRT rows come from one board, the PyTorch and ORT rows from a second one
with the same JetPack and clock settings, which runs X-VLA and EVO1 a few percent faster.
YMMV. `RAM in use` is the whole system while inferring.

## What every run logs

Each run writes one JSON to [`results/`](results/), and `python -m bench report` turns
them into [the results page](docs/RESULTS.md). Besides the latency distribution, a run
records:

- per-stage timings, and the first call after load;
- whole-board RAM and swap, process RSS, CPU and GPU load;
- power per rail and energy per inference;
- temperatures, and drift over a 5-minute sustained run;
- the clocks the board actually ran at (every CPU core, GPU, memory) next to the ones it
  was set to, plus over-current and thermal throttle events;
- action parity against the stock PyTorch policy;
- the board state: JetPack, power mode, package versions, repo commit.

[The profile](docs/06-profile.md) adds an Nsight Systems trace of every model: where each
stage's time goes and whether it is compute- or bandwidth-bound.

## Parity

Every pure TensorRT model reproduces the stock PyTorch FP32 actions to **cosine 0.99999
or better, within 0.27 % of the action range over the whole action chunk**.

Each bundle carries the stock policy's output for seeded inputs and noise, and loading
refuses to run if the engines miss it. Per-model values are in
[the results](docs/RESULTS.md#parity); `python -m bench parity` compares any two runs.

## Run it

Export on the machine you fine-tune on, copy the bundle to the Orin whole, and run it
there. Engines are built on the Orin on the first run and cached.

```bash
# on the export machine
export/setup.sh                       # SmolVLA, X-VLA and EVO1
export/export.sh lerobot/smolvla_base ~/bundles/smolvla-base-split --views 2
export/export.sh lerobot/xvla-base ~/bundles/xvla-base-split
export/export.sh zuoxingdong/evo1_libero ~/bundles/evo1-libero-split \
    --task "pick up the black bowl and place it on the plate"

export/setup.sh groot                 # GR00T N1.6
export/export.sh nvidia/GR00T-N1.6-3B ~/bundles/groot-n16-base-split
export/setup.sh groot17               # GR00T N1.7
export/export.sh nvidia/GR00T-N1.7-3B ~/bundles/groot-n17-base-split

# on the Orin
scripts/00_host_prep.sh
scripts/11_env_ort.sh
MODEL=smolvla-base   BUNDLE=~/bundles/smolvla-base-split   scripts/run_all.sh
MODEL=groot-n16-base BUNDLE=~/bundles/groot-n16-base-split scripts/run_all.sh
```

`run_all.sh` also runs the PyTorch reference for SmolVLA and X-VLA when its venv exists
(`scripts/10_env_torch.sh`, `scripts/13_env_torch_xvla.sh`) and the checkpoint has been
fetched (`python -m bench fetch --model <model> --what torch`).

- GR00T N1.7 reads its tokenizer from the gated
  [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B): accept
  its terms on Hugging Face before exporting. It sees every camera now and 30 frames
  earlier; the runtime reuses the earlier frame's encode.
- `evo1-libero` actions mean something for LIBERO's embodiment only.
- The ORT rows ran the Hugging Face bundles ([`eetmie/smolvla-base-onnx`](https://huggingface.co/eetmie/smolvla-base-onnx),
  [`eetmie/xvla-base-onnx`](https://huggingface.co/eetmie/xvla-base-onnx)) with
  `bench ort-split`. Building X-VLA's ORT engines needs 4 GB swap and a freshly booted,
  headless board ([host setup](docs/01-host-setup.md#swap--4-gb-and-build-on-a-freshly-booted-headless-board)).

`export/` also takes your own fine-tuned LeRobot checkpoint; see
[export/README.md](export/README.md).

## Documentation

- [Host setup](docs/01-host-setup.md)
- [Python environments](docs/02-environments.md)
- [Model and runtime contracts](docs/03-backends.md)
- [Metric definitions](docs/04-metrics.md)
- [Benchmark runbook](docs/05-runbook.md)
- [Profile: where the time goes](docs/06-profile.md)
- [Measured results](docs/RESULTS.md)

## Thanks

Big thanks to the teams behind [LeRobot and SmolVLA](https://github.com/huggingface/lerobot),
[X-VLA](https://thu-air-dream.github.io/X-VLA/), [EVO-1](https://github.com/MINT-SJTU/Evo-1),
[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T), and
[Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi) for
sharing their models and code. Thanks also to [FlashRT](https://github.com/flashrt-project/FlashRT)
for deployment and optimization ideas.

## Scope

Inference cost and parity on the board. No training, fine-tuning, robot control or
camera capture. TensorRT engines are built on the Jetson and never copied between
machines; the ONNX bundles are the portable artifacts.

Repository code is MIT. Model weights and derived exports keep their upstream licenses
(GR00T N1.6: NVIDIA One-Way Noncommercial License; N1.7: NVIDIA License with a
non-commercial use limitation); consult each model card before redistribution.
