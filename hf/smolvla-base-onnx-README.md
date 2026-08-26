---
license: apache-2.0
base_model: lerobot/smolvla_base
tags:
  - onnx
  - vla
  - robotics
  - jetson
  - tensorrt
  - smolvla
library_name: onnx
---

# SmolVLA 450M — split ONNX export (lerobot 0.5.1)

A nine-graph ONNX export of [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base),
cut so each graph builds a TensorRT engine inside the 8 GB of a Jetson Orin Nano, with
the flow-matching denoise loop run in Python. Weights are unchanged.

**Traced under lerobot 0.5.1.** That is not a footnote — see below.

Two camera slots (prefix 177 = 2x64 image + 48 language + 1 state), chunk 50, 10 denoise
steps. A padded slot costs no vision pass but still occupies its tokens, so a runtime
must feed the same slot count the export was built with.

## Measured on an Orin Nano Super (JetPack 7.2, clocks pinned, 2 real cameras)

| | |
|---|---|
| latency | **169.32 ms p50** (p95 170.23, std 1.00) = **5.90 Hz** |
| resident | 4990 MB |
| parity vs PyTorch | **cosine 0.9993839**, max abs 0.3439, 1.94% of commanded range |

Parity is cross-backend and on-device: this export at fp16/TensorRT against
`lerobot/smolvla_base` in PyTorch fp32, identical seeded observations and noise.

## Why a second SmolVLA export exists

[`ainekko/smolvla_base_onnx`](https://huggingface.co/ainekko/smolvla_base_onnx) came
first. Its weights are bit-identical to the checkpoint (all initializers match, 125
exactly and 73 under transpose), but under **lerobot 0.5.1** its actions land 12.92% of
commanded range from the PyTorch reference against this export's 1.94%. The divergence is
`smolvlm_vision` alone — cosine 0.822 on the image embedding, while `smolvlm_text` is
bit-identical. Not precision: the same graphs on the CPU EP at fp32 deviate identically.

Which export is "correct" depends on the lerobot version you consider canonical, and we
cannot answer that: both sides of our comparison were traced and scored under 0.5.1. If
you run lerobot 0.5.1, use this one. Full working in
[`docs/03-backends.md`](https://github.com/eetmie/jetson-orin-nano-vla/blob/main/docs/03-backends.md).

Runtime and benchmark harness: [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla).
