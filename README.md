# jetson-orin-nano-vla

Recipes and measurements for running public base VLA models on an **8 GB Jetson
Orin Nano Super**. The repository has two deployable base-model profiles and one
explicitly nondeployable EVO1 infrastructure profile:

| model | upstream checkpoint / initializer | split ONNX bundle |
|---|---|---|
| SmolVLA 450M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | [`eetmie/smolvla-base-onnx`](https://huggingface.co/eetmie/smolvla-base-onnx) |
| X-VLA 0.9B | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | [`eetmie/xvla-base-onnx`](https://huggingface.co/eetmie/xvla-base-onnx) |
| EVO1 775M bootstrap | [`OpenGVLab/InternVL3-1B-hf`](https://huggingface.co/OpenGVLab/InternVL3-1B-hf), pinned revision | local checksummed export; random action head |

## Measured fit

Retained runs use pinned MAXN_SUPER clocks and deterministic synthetic observations.
They measure inference cost, not robot-task quality.

| model / runtime | views | p50 | p95 | rate |
|---|---:|---:|---:|---:|
| SmolVLA PyTorch FP32 | 2 | 1167.93 ms | 1176.65 ms | 0.86 Hz |
| SmolVLA split ONNX FP16 | 2 | 189.89 ms | 190.93 ms | 5.25 Hz |
| X-VLA split ONNX FP16 | 3 | 415.94 ms | 418.01 ms | 2.40 Hz |
| EVO1 bootstrap split ONNX mixed FP16 | 1 | 289.18 ms | 292.08 ms | 3.46 Hz |

The split bundles fit because the large policies are divided into independently built
TensorRT engines. A whole-policy TensorRT build exceeds the board's unified-memory
budget. Full memory, power, CPU, thermal, validity, and per-graph measurements are in
[the generated results](docs/RESULTS.md).

## Parity gate

Speed is only worth measuring if the actions survive the conversion. On this board that
is a live question rather than a formality: compute 8.7 makes FP16 the only fast reduced
precision available, and a blanket FP16 cast is exactly what collapsed SmolVLA's SigLIP
vision tower to cosine 0.805 elsewhere. Every backend is therefore handed the *same*
seeded observations and the *same* injected noise (`bench/obs.py`), so the action chunks
line up element by element rather than only in distribution.

**The gate is two conditions, and both must hold:**

- `cosine >= 0.999` — direction
- `max_abs_diff <= 1%` of the **reference run's own observed action range** — amplitude

Cosine alone hides a scale error, and an absolute difference means nothing without a
range, so the difference is normalised against the range the reference policy actually
commands. That keeps the number comparable across policies with different action spaces.

The measured values are in [the results](docs/RESULTS.md#parity). The short version: the
converted models reproduce their reference actions to **cosine 0.9993 or better, and
within 0.49 % of the action range on the executed action**. Later steps in a long chunk
drift further, so a deployment that runs the whole horizon open-loop should look at the
full-chunk figure there too.

```bash
python -m bench parity results/smolvla-base.torch.json results/smolvla-base.ort.json \
    --reference smolvla-base.torch
```

It exits nonzero on a miss, and refuses any pair whose observations or injected noise
differ rather than reporting a cosine against a sequence the reference never saw. X-VLA
has no PyTorch counterpart run on this board yet, so it has no cross-backend value;
EVO1 has no deployable PyTorch reference at all and is checked against a native fixture
carried inside its bundle, which fails closed during load.

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

## Run the EVO1 infrastructure bootstrap

The current EVO1 artifact proves export, TensorRT execution, LeRobot 0.6.1 parity,
memory fit, and performance. Its action head is deterministic random initialization.
It is **not a trained policy and must never control a robot**. In particular, it is
not the `MINT-SJTU/Evo1_RoboTwin2_clean` trained checkpoint.

Export the bundle with the companion Spark workflow, copy the entire checksummed
directory to the Jetson, then run:

```bash
scripts/00_host_prep.sh
scripts/11_env_ort.sh

.venv-ort/bin/python scripts/check_evo1_fixture.py \
    --bundle ~/bundles/evo1-bootstrap-split \
    --cache-dir ~/.cache/jetson-orin-nano-vla/evo1-trt

.venv-ort/bin/python -m bench ort-split \
    --model evo1-bootstrap \
    --bundle ~/bundles/evo1-bootstrap-split \
    --cache-dir ~/.cache/jetson-orin-nano-vla/evo1-trt \
    --iters 100
```

The benchmark itself repeats the native-fixture check during load and fails closed if
parity is below the threshold. The standalone command is useful before a long run
because it prints every graph provider and boundary comparison.

The first ONNX run builds TensorRT engines serially and takes several minutes. Later
runs reuse the persistent cache. No camera is required: the benchmark defaults to
deterministic in-memory observations.

## Documentation

- [Host setup](docs/01-host-setup.md)
- [Python environments](docs/02-environments.md)
- [Model and runtime contracts](docs/03-backends.md)
- [Metric definitions](docs/04-metrics.md)
- [Benchmark runbook](docs/05-runbook.md)
- [Measured results](docs/RESULTS.md)

## Scope

This repository downloads, runs, and compares two public base checkpoints and the
nondeployable EVO1 export profile. It does not contain training, fine-tuning, robot
control, camera capture, or a trained EVO1 action head. TensorRT engines are built on
the Jetson and are never copied between machines; the ONNX bundles are the portable
artifacts.

Repository code is MIT. Model weights and derived exports retain their respective
upstream licenses; consult each model card before redistribution.
