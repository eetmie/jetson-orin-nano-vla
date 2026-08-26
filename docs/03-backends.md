# 3. Supported models and backends

This repository now has one reference path and one deployment path:

| backend | purpose |
|---|---|
| `torch` | numerical and performance reference using the LeRobot checkpoint |
| `ort-split` | split ONNX through ONNX Runtime, TensorRT, CUDA, and CPU providers |

The split layout is required on the 8 GB Orin Nano. Each heavy component builds and
loads independently, keeping TensorRT's temporary FP32 weight copies inside unified
memory. Whole-policy engine builds exceed that budget and are intentionally absent.

## Models

| key | family | params | Torch checkpoint | split bundle |
|---|---|---:|---|---|
| `smolvla-base` | SmolVLA | 450 M | `lerobot/smolvla_base` | `ainekko/smolvla_base_onnx` or a local export |
| `xvla-base` | X-VLA | 880 M | `lerobot/xvla-base` | local export |

Use `python -m bench models` for the exact shapes and defaults. A fine-tuned checkpoint
does not need a registry entry: pass `--family`, `--checkpoint`, and `--bundle`.

Base checkpoints are useful for latency and memory, not task quality. Parity must compare
the same weights, tokenizer, camera-slot layout, observation, and seeded noise.

## SmolVLA

The split contains nine graphs:

- vision, expert prefill, and expert decode are the heavy TensorRT graphs;
- text, state, action, and time projectors are small graphs;
- the flow-matching loop runs for ten steps by default.

The current harness moves the four per-step projectors to the GPU and uses IOBinding by
default. Both changes were validated as bit-identical to the earlier CPU/numpy path.
Use `--projectors cpu` or `--no-iobinding` only for a controlled A/B.

Camera slots are fixed at export time. `--views` is the number of real images;
`--cam-slots` is the export contract. An empty slot still occupies prefix tokens but
does not require a vision pass.

The public `ainekko` export and locally traced exports are not interchangeable parity
references across LeRobot versions. Always keep `export_info.json`, tokenizer files,
normalization stats, and the ONNX graphs together.

## X-VLA

The split contains twelve graphs around DaViT, BART conditioning, and the policy
transformer. X-VLA has no prefill/decode KV-cache seam: conditioning attends to action
tokens, so the policy stack reruns on every denoising step. Reducing the step count is
therefore the main latency lever, and it changes policy behaviour.

The checkpoint declares three image-view slots. Bundles can limit how many views receive
real images, but the sequence contract remains export-specific.

The in-repo PyTorch X-VLA reference currently stops before inference with
`Sequence length 1204 exceeds max_len_seq=512`; the split path uses 262 tokens. Until
that mismatch is resolved, X-VLA's older CPU parity result is supporting evidence, not
equivalent to the on-device cross-backend SmolVLA gate. The current audit is recorded in
[`07-audit-followups.md`](07-audit-followups.md).

## Getting and exporting bundles

```bash
python -m bench fetch --model smolvla-base --what both
python -m bench fetch --model xvla-base --what torch
```

There is no published X-VLA split bundle. Export it on the DGX Spark with LeRobot 0.6.1,
then copy the ONNX bundle—not TensorRT engines—to the Jetson:

```bash
CHECKPOINT=~/models/xvla-base \
OUT=~/bundles/xvla-base-split \
scripts/export_xvla_split.sh
```

Use FP16 X-VLA weights. The FP32 bundle exhausted the practical build budget before
finishing the first engine; the FP16 bundle built successfully. Engine caches belong to
the exact JetPack/TensorRT/GPU combination and remain local to the board.
