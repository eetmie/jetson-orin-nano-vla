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

The X-VLA reference now reads the saved LeRobot processor contract instead of the raw
config's stale 1024-token value; the matching 0.6.1 stack uses 50 language tokens.
Schema-v2 bundles carry exact tokenizer, checkpoint, dimensions, and normalization
identity. The 250-step digging smoke checkpoint passes full CPU FP32 parity at the
conditioning, padded 20-D model-action, normalized 4-D, and physical 4-D boundaries,
and its Jetson FP16 result passes the hardened gate on eight held-out recorded IR
observations. CPU ORT FP32 reaches cosine 1.000000 / 0.021% maximum range error;
Jetson TensorRT FP16 reaches cosine 0.9999763 / 0.394%, and repeats bit-identically in a
fresh process. See
[`results/xvla-digging-contract/`](../results/xvla-digging-contract/).
The current audit is recorded in
[`07-audit-followups.md`](07-audit-followups.md).

`--xvla-iobinding` keeps the conditioning tensor and the three denoiser split
intermediates on CUDA. The paired 40-inference Jetson A/B was bit-identical and reduced
p50 by 1.0% and p95 by 2.4%, so it remains an explicit, measured opt-in rather than the
headline default. With an ordinary bundle, the host still performs X-VLA's interpolation
between denoising steps.

An experimental export made with `--fuse-denoise-interpolation` changes `denoise_0` to
consume fixed noise plus the previous action and form the exact interpolation inside the
graph. With `--xvla-iobinding`, noise, action, conditioning, and all split intermediates
then remain on CUDA for the entire ten-step loop. It passes real-IR Torch parity and all
12 graphs have measured TensorRT node events, but its paired 40-call p50 is 340.55 ms
versus 339.28 ms for partial IOBinding. Keep it experimental: removing the final host
round-trip did not produce a material latency win on this stack.

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
the exact JetPack/TensorRT/GPU combination and remain local to the board. X-VLA writes
an engine-cache manifest after the first complete prebuild. Later loads verify bundle,
precision, builder settings, ORT, TensorRT, CUDA, L4T, hardware, and the exact cached
file inventory before skipping the twelve validation subprocesses. A stale, truncated,
or mixed cache fails closed; use a new cache directory instead of reusing it.

On a machine where `TensorrtExecutionProvider` is unavailable, the X-VLA adapter skips
engine prebuild and can run the same graphs through CPU ORT with `--precision fp32`.
That path is for export-correctness parity, not Jetson performance.
