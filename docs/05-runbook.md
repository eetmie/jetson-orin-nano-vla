# 5. Base-model benchmark runbook

Run one model process at a time because CPU and GPU share the same 8 GB memory pool.

## 1. Prepare the board

```bash
scripts/00_host_prep.sh
scripts/00_host_prep.sh --verify
```

The expected state is MAXN_SUPER with pinned CPU, GPU, and EMC clocks, working swap,
and a persistent TensorRT engine cache.

## 2. Build the environments

For SmolVLA:

```bash
scripts/10_env_torch.sh
scripts/11_env_ort.sh
```

For X-VLA:

```bash
scripts/13_env_torch_xvla.sh
scripts/11_env_ort.sh
```

Verify the measurement plumbing before trusting a result:

```bash
.venv-torch/bin/python -m bench selftest --seconds 4
```

## 3. Download one base model

```bash
hf auth login  # required only while eetmie/* is private
scripts/fetch_models.sh smolvla-base
# or:
scripts/fetch_models.sh xvla-base
```

## 4. Run the complete recipe

```bash
MODEL=smolvla-base scripts/run_all.sh
MODEL=xvla-base scripts/run_all.sh
```

The script records a PyTorch FP32 reference, the split ONNX FP16 deployment, and a
sustained thermal run when the required environments and artifacts are present. Set
`SUSTAINED=0` to skip the five-minute sustained pass.

The default observation is deterministic and in memory, so no camera or robot hardware
is needed.

## 5. Run one backend

```bash
M=smolvla-base

.venv-torch/bin/python -m bench torch \
    --model "$M" --checkpoint ~/bundles/$M-torch \
    --iters 30

.venv-ort/bin/python -m bench ort-split \
    --model "$M" --bundle ~/bundles/$M-split \
    --views 2 --iters 100
```

For X-VLA, use `.venv-torch-xvla` and `--views 3`.

## 6. Compare and report

```bash
python -m bench parity results/smolvla-base.torch.json results/smolvla-base.ort.json \\
    --reference smolvla-base.torch
python -m bench report results --out docs/RESULTS.md
```

Every result JSON records latency, memory, CPU, power, thermals, board state, runtime
versions, and enough saved actions for a like-for-like parity check.
