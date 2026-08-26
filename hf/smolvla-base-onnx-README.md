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
LeRobot 0.5.1. The split lets TensorRT build and run the model within an 8 GB Jetson
Orin Nano; weights are unchanged.

Contract: two camera slots, 177-token prefix, 50-action chunk, and ten denoise steps.
The runtime must use the tokenizer and normalization statistics shipped with the bundle.

## Retained Orin Nano result

Pinned clocks, JetPack 7.2, two real views, FP16 TensorRT:

| p50 | p95 | rate | parity vs Torch FP32 |
|---:|---:|---:|---:|
| 169.32 ms | 170.23 ms | 5.90 Hz | cosine 0.9993839; 1.94% of action range |

Parity uses identical observations and seeded noise. It is useful evidence, but misses
the benchmark repo's current strict 1% range gate. Base weights are intended for runtime
measurement, not robot task performance.

Runtime, limitations, and reproducible benchmark:
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla).
