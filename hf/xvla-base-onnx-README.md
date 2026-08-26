---
license: apache-2.0
base_model: lerobot/xvla-base
tags:
  - onnx
  - vla
  - robotics
  - jetson
  - tensorrt
  - x-vla
library_name: onnx
---

# X-VLA 0.9B — split ONNX export (lerobot 0.6.1)

A twelve-graph ONNX export of [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base),
cut so that each graph builds a TensorRT engine inside the 8 GB of a Jetson Orin Nano.
Weights are unchanged. No other public X-VLA ONNX export exists.

**Traced under lerobot 0.6.1**, whose X-VLA sits on transformers' Florence-2. lerobot
0.5.1 vendors its own Florence-2 with a different module layout and cannot produce this
export.

Budgeted at 0.40 GB of fp32 weights per graph — that is what keeps each TensorRT build
inside 8 GB. Three declared image views, chunk 30, 10 denoise steps.

## Measured on an Orin Nano Super (JetPack 7.2, clocks pinned)

| views | latency | resident |
|---|---|---|
| 3 (native) | **415.94 ms p50** (p95 418.01) = **2.40 Hz** | 4741 MB |
| 1 | **343.22 ms p50** (p95 345.11) = **2.91 Hz** | 4524 MB |

All 12 graphs run on TensorRT; denoise is 71% of wall and cannot be KV-cached, because
the policy transformer is bidirectional — conditioning attends to action tokens, so every
block re-runs each step. `num_denoising_steps` is therefore the only real latency lever.

Parity vs the PyTorch reference: **cosine 0.999993** (action), 0.999953 (cond_tokens).
Measured by `xvla-runtime/parity.py`, which builds both sides in one script at CPU fp32 —
a **less demanding test** than the cross-backend comparison used for the SmolVLA export,
and quoted as such. A like-for-like on-device comparison is not currently possible: the
PyTorch path fails before inference with `Sequence length 1204 exceeds max_len_seq=512`.

## Use the FP16 weights

The fp32 bundle reaches 6 GB resident during the TensorRT build on a board with the stock
2 GB swap and completes zero engines. Halving the weights (LayerNorm and Softmax kept
fp32) costs cosine 1.000000 -> 0.999993 and builds cleanly.

Runtime and benchmark harness: [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla).
