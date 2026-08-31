# jetson-orin-nano-vla

A focused benchmark and deployment harness for running VLA policies on an **8 GB
Jetson Orin Nano Super**.

The supported runtime matrix is deliberately small:

| model | PyTorch reference | Jetson deployment |
|---|---|---|
| `smolvla-base` (450 M) | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | split ONNX on ONNX Runtime + TensorRT |
| `xvla-base` (880 M) | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | split ONNX on ONNX Runtime + TensorRT |

The benchmark records latency, parity, process and system memory, CPU use, board power,
thermals, and per-graph time. Unsplit whole-policy TensorRT builds are not supported:
they exceed the board's unified-memory budget.

## Current speeds

Latest retained measurements on the pinned Orin Nano, using in-memory observations:

| path | views | p50 ms | p95 ms | Hz | status |
|---|---:|---:|---:|---:|---|
| SmolVLA PyTorch FP32 | 2 | 1167.93 | 1176.65 | 0.856 | numerical reference |
| SmolVLA split FP16 | 2 | 189.89 | 190.93 | 5.248 | unmodified export |
| SmolVLA split FP16, vision candidate | 2 | **164.66** | **165.88** | **6.044** | experimental |
| X-VLA split FP16 | 3 | 415.94 | 418.01 | 2.40 | retained base baseline |
| X-VLA digging smoke split FP16 | 1 | **348.81** | **353.74** | **2.87** | parity + placement PASS |
| X-VLA digging smoke, real IR fixture | 1 | 359.92 | 365.97 | 2.79 | real-input parity PASS |

The fine-tuned X-VLA rows are a mechanical deployment of the 250-step smoke checkpoint,
not a task-quality claim. On eight held-out recorded IR observations, TensorRT is 5.8×
faster than its Jetson PyTorch FP32 reference and passes the physical-action gate; see
[`results/xvla-digging-contract/`](results/xvla-digging-contract/).
A fresh TensorRT process is bit-identical. CPU ORT FP32 also passes against Torch, and
one non-headline ORT profile confirms measured TensorRT execution for all 12 split graphs
with no CUDA/CPU fallback node events.

The SmolVLA candidate removes redundant vision NaN guards. It is repeatable and faster,
but is not the default: its FP32-reference parity is cosine 0.9989535 and 1.566% of
action range, outside this repo's strict 0.999 / 1% gate. See
[`results/audit-patch-smolvla/`](results/audit-patch-smolvla/) for the signed evidence.

On the current fine-tuned `kaivuriprokkis` one-IR pipeline, the model call measured
201.94 ms p50 / 210.21 ms p95; the complete live inference loop is about 230 ms. That
deployment measurement is useful context, but is separate from the retained synthetic
benchmark rows above.

## Quickstart

```bash
scripts/00_host_prep.sh
scripts/10_env_torch.sh
scripts/13_env_torch_xvla.sh    # only for X-VLA
scripts/11_env_ort.sh

.venv-torch/bin/python -m bench selftest
scripts/fetch_models.sh smolvla-base
MODEL=smolvla-base scripts/run_all.sh
```

Run one backend:

```bash
python -m bench torch --model smolvla-base \
    --checkpoint ~/bundles/smolvla-base-torch --weights float32 --iters 30

python -m bench ort-split --model smolvla-base \
    --bundle ~/bundles/smolvla-base-split --views 2 --iters 100

python -m bench parity results --reference smolvla-base.torch-fp32
python -m bench report results --out docs/RESULTS.md
```

A local fine-tuned policy uses the same path:

```bash
python -m bench ort-split --family smolvla --bundle ~/bundles/my-split-export \
    --state-dim 3 --action-dim 4 --label mine.ort-split
```

## Documentation

- [Host setup](docs/01-host-setup.md): board state, swap, and engine cache
- [Environments](docs/02-environments.md): the three isolated Python environments
- [Backends](docs/03-backends.md): supported models, runtime contracts, and exports
- [Metrics](docs/04-metrics.md): how results and parity are interpreted
- [Runbook](docs/05-runbook.md): a complete benchmark pass
- [Optimization status](docs/06-optimization-backlog.md): retained wins and open gates
- [Audit log](docs/07-audit-followups.md): detailed findings and Spark → Jetson handoff
- [Generated results](docs/RESULTS.md): summary of committed run JSONs

## Scope and licence

This repository benchmarks model inference. It does not open cameras or drive a robot,
so capture, USB, control, and logging overhead must be measured on the target system.
TensorRT engines are built on the Jetson and are never copied between machines; ONNX
bundles are the portable artifacts.

Repository code is MIT. Model weights and derived exports remain under their upstream
Apache 2.0 licences.
