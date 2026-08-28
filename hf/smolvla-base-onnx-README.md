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

Pinned clocks (MAXN_SUPER), JetPack R39.2.1, both camera slots fed a deterministic
procedural scene, FP16 with GPU projectors and IOBinding, ONNX Runtime 1.24.0 on
TensorRT 10.16.2.10 — seven graphs on TensorRT, two on CPU:

| p50 | p95 | rate | resident |
|---:|---:|---:|---:|
| 189.89 ms | 190.93 ms | 5.25 Hz | 1871 MB |

Vision is 91.2 ms of that and the denoise decode 59.3 ms.

## Parity

Against a LeRobot 0.5.1 PyTorch FP32 reference on the same board, from identical
observations and identical injected flow-matching noise. Eight action chunks over the
six declared action dimensions, as a percentage of the reference run's action range:

| cosine min | median | mean | p99 | max |
|---:|---:|---:|---:|---:|
| 0.9993415 | 0.103% | 0.184% | 0.894% | 2.052% |

Read the maximum as a maximum. It is the single worst element of 2400; half of them
differ by less than 0.103% of range.

Both runs use the benchmark's procedural observation source, not camera frames. That is
sound for comparing two runtimes on identical input, but a procedural scene is out of
distribution for the policy, and on this benchmark's own evidence synthetic input can
understate FP16 drift by roughly 3x relative to real frames. Treat these figures as a
floor for a like-for-like runtime comparison, not as a field measurement.

Base weights are intended for runtime measurement, not robot task performance.

Runtime, limitations, and reproducible benchmark:
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla).
