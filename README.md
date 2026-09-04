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
is a live question: compute 8.7 makes FP16 the only fast reduced precision available, and
a blanket FP16 cast is exactly what collapsed SmolVLA's SigLIP vision tower to cosine
0.805 elsewhere. So every backend is handed the *same* seeded observations and the *same*
injected noise (`bench/obs.py`), which makes the saved action chunks comparable element by
element rather than only in distribution.

**The gate is two conditions, and both must hold** (`bench/parity.py`):

- `cosine >= 0.999` — direction
- `max_abs_diff <= 1%` of the **reference run's own observed action range** — amplitude

Cosine alone hides a scale error, and an absolute difference means nothing without a
range, so the difference is normalised against the range the reference policy actually
commands. That keeps the number comparable across policies with different action spaces.

### Measured, base models only

| comparison | cosine min | max abs diff | % of ref range | verdict |
|---|---:|---:|---:|:--|
| SmolVLA split ONNX FP16 vs PyTorch FP32 | 0.999339 | 0.3641 | 2.05 % | **FAIL** |
| EVO1 bootstrap split ONNX vs native fixture | 0.999991 | 0.00747 | — | pass |
| X-VLA split ONNX FP16 | — | — | — | not measured |

```bash
python -m bench parity results/smolvla-base.torch.json results/smolvla-base.ort.json \
    --reference smolvla-base.torch
```

**SmolVLA fails on amplitude, not on direction.** Cosine clears its bar comfortably
(0.999339 min, 0.999723 mean across 8 observations); the 1%-of-range bar is what it
misses, at 2.05 %. Two things narrow what that means:

- The difference is entirely in the **6 real action dimensions**. The 26 padding dimensions
  of the 32-wide vector agree to ~0.007, so this is not an artefact of comparing padding.
- It is **not uniform across the chunk**. The first action — the one a control loop actually
  executes — differs by 0.086, or 0.49 % of range, inside the gate. The 0.364 excursion is
  at chunk steps 7-8, deep in the 50-step horizon.

Read it as: base SmolVLA FP16 is directionally sound and fine to benchmark, but the far end
of a long chunk drifts enough that open-loop execution of the whole horizon is not certified
by this measurement. Any fine-tuned bundle must re-run the gate on its own export.

EVO1's gate is a different kind: the native fixture ships inside the bundle, is verified
during load, and **fails closed** before the benchmark runs. Its boundaries measure
0.999606 vision, 0.999546 fused valid tokens, 0.999991 action.

X-VLA has no PyTorch counterpart run on this board, so there is nothing here to compare its
FP16 split against; its export parity was established off-board and is not this repo's claim.

`docs/RESULTS.md` shows `not_checked` for the SmolVLA parity column because that column
records validity captured *at run time*; the cross-backend comparison above is a separate
step run after both results exist.

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
