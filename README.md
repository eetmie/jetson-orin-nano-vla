# jetson-orin-nano-vla

Recipes and measurements for running VLA models on an **8 GB Jetson Orin Nano Super**:
four public base checkpoints and the trained EVO1 LIBERO and π0.5 LIBERO checkpoints,
all on a pure TensorRT runtime (`bench trt-split`).

**Tested on JetPack 7.2.1 (L4T R39.2.1).**

| model | upstream checkpoint |
|---|---|
| SmolVLA 450M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| EVO1 775M LIBERO | [`zuoxingdong/evo1_libero`](https://huggingface.co/zuoxingdong/evo1_libero) |
| X-VLA 0.9B | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) |
| GR00T N1.7 3B | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| GR00T N1.6 3B | [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) |
| π0.5 LIBERO 3.3B (prototype) | [openpi](https://github.com/Physical-Intelligence/openpi) `pi05_libero` |

## Measured fit

Pinned MAXN_SUPER clocks, deterministic synthetic observations, default denoising steps.
These measure inference cost, not robot-task quality.

| model / runtime | views | p50 | p95 | rate | RAM in use |
|---|---:|---:|---:|---:|---:|
| **SmolVLA** TensorRT + AOT Triton, mixed FP16, lean runtime | 2 | 104.60 ms | 105.73 ms | 9.55 Hz | 1.65 GB |
| EVO1 LIBERO split, ORT mixed FP16 | 2 | 414.67 ms | 424.72 ms | 2.41 Hz | 6.00 GB |
| **EVO1 LIBERO** TensorRT + AOT Triton, mixed FP16 | 2 | 333.37 ms | 335.03 ms | 3.00 Hz | 2.62 GB |
| X-VLA PyTorch FP32 | 3 | 2313.50 ms | 2320.89 ms | 0.43 Hz | 5.45 GB |
| X-VLA split, ORT FP16 | 3 | 391.55 ms | 407.33 ms | 2.55 Hz | 5.39 GB |
| **X-VLA** TensorRT + AOT Triton, mixed FP16 | 3 | 342.31 ms | 343.15 ms | 2.92 Hz | 2.93 GB |
| **GR00T N1.7 3B** TensorRT + AOT Triton, mixed FP16 | 3 (×2 frames) | 272.39 ms | 273.77 ms | 3.67 Hz | 5.72 GB |
| **GR00T N1.6 3B** TensorRT + AOT Triton, mixed FP16 | 3 | 272.49 ms | 273.47 ms | 3.67 Hz | 5.30 GB |
| π0.5 LIBERO split, pure TensorRT FP16, padded prefix | 2 | 684.93 ms | 686.16 ms | 1.46 Hz | 6.34 GB |
| **π0.5 LIBERO** TensorRT + AOT Triton FP16, compact prefix | 2 | 436.30 ms | 438.01 ms | 2.29 Hz | 6.30 GB |

`RAM in use` is whole-system usage during inference. Bold rows are 300-second runs
with FP32 accumulation ([all-model evidence](results/all-models-20261010/fp32-accumulate/),
[GR00T](results/groot-triton-20261012/), [other models](results/models-20261012/)).
The PyTorch and ORT rows use a second board with the same JetPack and clocks, which
runs X-VLA and EVO1 a few percent faster. π0.5's padded-prefix row is an earlier run.
See the [optimization playbook](docs/07-optimization-playbook.md) for kernel choices
and speed, RAM and precision tradeoffs.

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
or better, within 0.27 % of the action range over the whole action chunk**. π0.5 is
checked the same way against its FP16 PyTorch conversion, which itself sits at cosine
0.99998 from the original mixed-precision policy.

Each bundle carries the stock policy's output for seeded inputs and noise, and loading
refuses to run if the engines miss it. Per-model values are in
[the results](docs/RESULTS.md#parity); `python -m bench parity` compares any two runs.

## Run it

Export on a Linux Python 3.12 machine with enough memory for the stock policy, copy
each whole bundle to the Orin, then build and benchmark there. The
[all-model workflow](docs/10-all-model-setup.md) provides all six export commands,
dedicated runtime/compiler environments and pinned model, source and dependency
versions.

```bash
# on the export machine
export/setup.sh smolvla
export/export.sh lerobot/smolvla_base ~/vla-bundles/smolvla-base \
    --views 2 --hoist-cross-kv

# on the Orin
scripts/00_host_prep.sh
scripts/12_env_trt.sh
scripts/14_env_triton.sh
python3 -m bench.tools.vla_setup --model smolvla-base \
    --bundle ~/vla-bundles/smolvla-base --work-dir ~/vla-runs/smolvla-new

# With all six bundles in subdirectories named by model key:
python3 -m bench.tools.vla_setup --model all --bundles-dir ~/vla-bundles \
    --work-dir ~/vla-runs/all-new --keep-going
```

Environment and work directories must be new. The workflow verifies bundles and
action parity, builds engines and runs models serially for 300 seconds each.
Use `--duration-s 10` for a smoke test. Inference needs neither Torch nor Triton.
See the [test audit](docs/09-test-audit.md) for validation coverage and the
[export guide](export/README.md) for fine-tuned LeRobot checkpoints.

## Documentation

- [Host setup](docs/01-host-setup.md)
- [Python environments](docs/02-environments.md)
- [Benchmark all six models](docs/10-all-model-setup.md)
- [Optimized SmolVLA setup](docs/08-smolvla-setup.md)
- [Test audit and clean-install coverage](docs/09-test-audit.md)
- [Model and runtime contracts](docs/03-backends.md)
- [Metric definitions](docs/04-metrics.md)
- [Benchmark runbook](docs/05-runbook.md)
- [Profile: where the time goes](docs/06-profile.md)
- [Optimization playbook](docs/07-optimization-playbook.md)
- [Measured results](docs/RESULTS.md)

## Thanks

Big thanks to the teams behind [LeRobot and SmolVLA](https://github.com/huggingface/lerobot),
[X-VLA](https://thu-air-dream.github.io/X-VLA/), [EVO-1](https://github.com/MINT-SJTU/Evo-1),
[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T), and
[Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi) for
sharing their models and code. Thanks also to [FlashRT](https://github.com/flashrt-project/FlashRT)
for deployment and optimization ideas, and to Jeremy Gracey's
[triton-kernel-lab](https://github.com/JeremyGracey-AI/triton-kernel-lab) for the
Orin Nano softmax examples used in the [initial Triton experiments](experiments/smolvla_triton/).

## Scope

Inference cost and parity on the board. No training, fine-tuning, robot control or
camera capture. TensorRT engines are built on the Jetson and never copied between
machines; the ONNX bundles are the portable artifacts.

Repository code is MIT. Model weights and derived exports keep their upstream licenses
(GR00T N1.6: NVIDIA One-Way Noncommercial License; N1.7: NVIDIA License with a
non-commercial use limitation); consult each model card before redistribution.
