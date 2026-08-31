# jetson-orin-nano-vla

Recipes and measurements for running two public base VLA models on an **8 GB Jetson
Orin Nano Super**. The repository intentionally supports only these models:

| model | upstream checkpoint | split ONNX bundle |
|---|---|---|
| SmolVLA 450M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | [`eetmie/smolvla-base-onnx`](https://huggingface.co/eetmie/smolvla-base-onnx) |
| X-VLA 0.9B | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | [`eetmie/xvla-base-onnx`](https://huggingface.co/eetmie/xvla-base-onnx) |

The two split repositories are currently private and require `hf auth login`; the same
links will work without authentication when they are made public. Their bundle metadata
and graph manifests have been checked against the runtime contracts in this repository.

## Measured fit

Retained base-model runs use pinned MAXN_SUPER clocks and deterministic synthetic
observations. They measure inference cost, not robot-task quality.

| model / runtime | views | p50 | p95 | rate |
|---|---:|---:|---:|---:|
| SmolVLA PyTorch FP32 | 2 | 1167.93 ms | 1176.65 ms | 0.86 Hz |
| SmolVLA split ONNX FP16 | 2 | 189.89 ms | 190.93 ms | 5.25 Hz |
| X-VLA split ONNX FP16 | 3 | 415.94 ms | 418.01 ms | 2.40 Hz |

Both ONNX bundles fit because the large policies are split into independently built
TensorRT engines. A whole-policy TensorRT build exceeds the board's unified-memory
budget. Full memory, power, CPU, thermal, and per-graph measurements are in
[the generated results](docs/RESULTS.md).

## Run SmolVLA base

```bash
scripts/00_host_prep.sh
scripts/10_env_torch.sh
scripts/11_env_ort.sh
scripts/fetch_models.sh smolvla-base

MODEL=smolvla-base scripts/run_all.sh
```

## Run X-VLA base

```bash
scripts/00_host_prep.sh
scripts/13_env_torch_xvla.sh
scripts/11_env_ort.sh
scripts/fetch_models.sh xvla-base

MODEL=xvla-base scripts/run_all.sh
```

The first ONNX run builds TensorRT engines serially and takes several minutes. Later
runs reuse the persistent cache. No camera is required: the benchmark defaults to
deterministic in-memory observations.

Run only the deployment path with:

```bash
.venv-ort/bin/python -m bench ort-split \
    --model smolvla-base \
    --bundle ~/bundles/smolvla-base-split \
    --iters 100
```

## Documentation

- [Host setup](docs/01-host-setup.md)
- [Python environments](docs/02-environments.md)
- [Model and runtime contracts](docs/03-backends.md)
- [Metric definitions](docs/04-metrics.md)
- [Benchmark runbook](docs/05-runbook.md)
- [Base-model results](docs/RESULTS.md)

## Scope

This repository downloads, runs, and compares the two base checkpoints. It does not
contain training, fine-tuning, robot control, camera capture, task-specific checkpoints,
or experimental model variants. TensorRT engines are built on the Jetson and are never
copied between machines; the ONNX bundles are the portable artifacts.

Repository code is MIT. Model weights and derived exports retain their upstream
Apache-2.0 licenses.
