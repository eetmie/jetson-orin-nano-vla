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

# X-VLA 0.9B — split ONNX export

A twelve-graph ONNX export of [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base),
cut so that each graph builds a TensorRT engine inside the 8 GB of a Jetson Orin Nano.
Weights are unchanged.

The equivalent for SmolVLA is [`ainekko/smolvla_base_onnx`](https://huggingface.co/ainekko/smolvla_base_onnx).
This is that, for X-VLA.

A reference runtime and a benchmark harness live in
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla):
