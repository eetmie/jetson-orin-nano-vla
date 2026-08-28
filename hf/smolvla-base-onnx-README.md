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

# SmolVLA 450M split ONNX

Nine ONNX graphs derived from
[`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base), traced with
LeRobot 0.5.1 at opset 17. The split lets TensorRT build and run the model within an
8 GB Jetson Orin Nano; weights are unchanged.

Contract: two camera slots at 512x512, a 177-token prefix, a 48-token language budget,
a 50-action chunk, and ten denoise steps. The runtime must use the tokenizer and
normalization statistics shipped with the bundle.

## Retained Orin Nano result

Pinned clocks (MAXN_SUPER), JetPack R39.2.1, two real camera views, FP16 with GPU
projectors and IOBinding, ONNX Runtime 1.24.0 on TensorRT 10.16.2.10 — seven graphs on
TensorRT, two on CPU:

| p50 | p95 | rate | resident |
|---:|---:|---:|---:|
| 189.89 ms | 190.93 ms | 5.25 Hz | 1871 MB |

Vision is 91.2 ms of that and the denoise decode 59.3 ms.

## Parity

Against a LeRobot 0.5.1 PyTorch FP32 reference on the same board, using identical
observations and seeded noise:

| cosine minimum | max absolute difference | as % of action range |
|---:|---:|---:|
| 0.999339 | 0.3641 | 2.052% |

Base weights are intended for runtime measurement, not robot task performance.

Runtime, limitations, and reproducible benchmark:
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla).
