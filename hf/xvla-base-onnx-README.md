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

Contract: three image-view slots, 30-action chunk, and ten denoise steps. X-VLA's
bidirectional policy transformer reruns on every denoise step; there is no KV-cache
prefill/decode seam.

## Retained Orin Nano result

Pinned clocks, JetPack 7.2, three views, FP16 TensorRT:

| p50 | p95 | rate | resident |
|---:|---:|---:|---:|
| 415.94 ms | 418.01 ms | 2.40 Hz | 4741 MB |

Older export-time CPU parity measured cosine 0.999993 for actions. This is less demanding
than an on-device cross-backend test: the current PyTorch harness hits a sequence-length
mismatch before inference. Treat the number as supporting evidence until that audit is
complete.

Runtime, limitations, and reproducible benchmark:
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla).
