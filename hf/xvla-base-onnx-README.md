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

# X-VLA 0.9B split ONNX

Twelve ONNX graphs derived from
[`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base), traced with LeRobot
0.6.1. The split and FP16 weights keep individual TensorRT builds within an 8 GB Jetson
Orin Nano.

Contract: three image-view slots at 50 tokens each, a 50-token language budget, a
30-action chunk, ten denoise steps, and 20-dimensional state and action vectors in
`ee6d` action mode. X-VLA's bidirectional policy transformer reruns on every denoise
step; there is no KV-cache prefill/decode seam.

## Retained Orin Nano result

Pinned clocks (MAXN_SUPER), JetPack R39.2.1, all three view slots fed a deterministic
procedural scene, FP16, ONNX Runtime 1.24.0 on TensorRT 10.16.2.10 — all twelve graphs on
TensorRT:

| p50 | p95 | rate | resident |
|---:|---:|---:|---:|
| 415.94 ms | 418.01 ms | 2.40 Hz | 4741 MB |

Denoising is 295.5 ms of that and vision 111.8 ms.

## Parity

Export-time CPU parity measured cosine 0.999993 for actions. There is no on-device
cross-backend measurement yet: the PyTorch reference harness hits a sequence-length
mismatch before inference. Treat the export-time figure as supporting evidence until
that audit is complete.

The latency row above also comes from the procedural observation source rather than
camera frames — sound for timing, since the transformer does the same work whatever the
pixels are, but it says nothing about predicted action values.

Base weights are intended for runtime measurement, not robot task performance.

Runtime, limitations, and reproducible benchmark:
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla).
