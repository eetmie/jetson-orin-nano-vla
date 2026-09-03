# 5. Benchmark runbook

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

EVO1 needs only the ONNX environment:

```bash
scripts/11_env_ort.sh
```

Verify the measurement plumbing before trusting a result:

```bash
.venv-ort/bin/python -m bench selftest --seconds 4
```

## 3. Obtain one model

For a public base:

```bash
hf auth login  # required only while eetmie/* is private
scripts/fetch_models.sh smolvla-base
# or:
scripts/fetch_models.sh xvla-base
```

For EVO1, copy the complete companion Spark export to a stable directory such as
`~/bundles/evo1-bootstrap-split`. Do not copy a TensorRT engine cache from another
machine. The current bundle is nondeployable and has a random action head.

## 4. Validate EVO1 before a long run

```bash
.venv-ort/bin/python scripts/check_evo1_fixture.py \
    --bundle ~/bundles/evo1-bootstrap-split \
    --cache-dir ~/.cache/jetson-orin-nano-vla/evo1-trt
```

The command verifies the entire manifest, builds/loads each engine serially, compares
the eleven-graph outputs to the native LeRobot 0.6.1 fixture, and exits nonzero below
the cosine threshold. `bench ort-split` repeats the same gate during load, so it cannot
record an apparently fast result after a parity failure.

## 5. Run the complete recipe

```bash
MODEL=smolvla-base scripts/run_all.sh
MODEL=xvla-base scripts/run_all.sh
MODEL=evo1-bootstrap \
    BUNDLE=~/bundles/evo1-bootstrap-split \
    scripts/run_all.sh
```

The public-base recipes record a PyTorch FP32 reference, the split ONNX FP16
deployment, and a sustained thermal run. EVO1 skips PyTorch because its reference is
the embedded native fixture. Set `SUSTAINED=0` to skip the five-minute sustained pass.

The default observation is deterministic and in memory, so no camera or robot hardware
is needed. EVO1 synthetic observations use the policy's native uniform `[-1, 1]` flow
noise; SmolVLA and X-VLA retain their normal noise.

## 6. Run one backend

```bash
M=smolvla-base

.venv-torch/bin/python -m bench torch \
    --model "$M" --checkpoint ~/bundles/$M-torch \
    --iters 30

.venv-ort/bin/python -m bench ort-split \
    --model "$M" --bundle ~/bundles/$M-split \
    --views 2 --iters 100
```

For X-VLA, use `.venv-torch-xvla` and `--views 3`. For EVO1:

```bash
.venv-ort/bin/python -m bench ort-split \
    --model evo1-bootstrap \
    --bundle ~/bundles/evo1-bootstrap-split \
    --cache-dir ~/.cache/jetson-orin-nano-vla/evo1-trt \
    --views 1 --iters 100
```

Do not override EVO1's 32 flow steps for a retained comparison: a different step count
does not match the embedded native action and fails the automatic parity gate.

## 7. Generate the report

```bash
python -m bench report results --out docs/RESULTS.md
```

Every result JSON records validity, latency, memory, CPU, power, thermals, board state,
runtime versions, and representative saved action chunks. EVO1 results also embed the
native-fixture comparison and the nondeployable/random-head safety flags.
