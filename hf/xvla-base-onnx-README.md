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

## Why it is split, and why into twelve

TensorRT imports weights as FP32 working copies regardless of the ONNX dtype, so the
build peak tracks the weight slice a single engine carries. Measured on an Orin Nano
8 GB:

```
build peak RSS  ≈  3.18 GB  +  5.63 × (FP32 weight GB)
```

which leaves room for roughly **0.40 GB of FP32 weights (~100 M params) per engine**.
All three of X-VLA's heavy components exceed that on their own, so each is split by
parameter budget:

| component | FP32 | engines |
|---|---:|---:|
| DaViT vision tower + projector | 1.44 GB | 4 |
| BART encoder + token embedding | 0.83 GB | 3 |
| policy transformer (24 blocks) | 1.21 GB | 4 |
| conditioning projections | 0.01 GB | 1 |

A monolithic export of a model this size does not build on an 8 GB board at any
precision — the floor is the weight import, not the node count.

## The graphs

| graph | params (M) | MB |
|---|---:|---:|
| `vision_0` | 89.6 | 358 |
| `vision_1` | 75.7 | 303 |
| `vision_2` | 94.6 | 378 |
| `vision_3` | 103.0 | 412 |
| `text_encoder_0` | 94.5 | 378 |
| `text_encoder_1` | 88.2 | 353 |
| `text_encoder_2` | 25.2 | 101 |
| `cond` | 2.1 | 9 |
| `denoise_0` | 75.6 | 303 |
| `denoise_1` | 75.6 | 302 |
| `denoise_2` | 75.6 | 302 |
| `denoise_3` | 75.6 | 302 |
| **total** | **875.1** | **3503** |

Constants in `bundle.json`: `num_image_views` 3, `valid_views`
1, `tokens_per_view` 50, `lang_len` 50,
`chunk_size` 30, `hidden_size` 1024, `dim_time`
32, `max_state_dim` 20, `num_denoising_steps`
10, `action_mode` `ee6d`, `domain_id`
0.

The four `denoise_*` graphs are 6 transformer blocks each and run **once per denoising
step** — ten times per action chunk at the default budget, over a 262-token sequence.
That is where the latency lives.

## Layout

`bundle.json` is the manifest: graph names, files, input/output names, parameter counts,
and the shape constants a runtime needs (`num_image_views`, `valid_views`,
`tokens_per_view`, `lang_len`, `chunk_size`, `hidden_size`, `dim_time`, `max_state_dim`,
`num_denoising_steps`, `action_mode`, `domain_id`).

Graphs run as: `vision_* → text_encoder_* → cond` once per observation, then `denoise_*`
once per denoising step over the cached conditioning.

## Two things to know before writing a runtime against this

**No KV cache is possible.** Unlike SmolVLA's prefill/decode split, X-VLA's policy
transformer is a bidirectional encoder over one concatenated sequence: the conditioning
tokens attend *to* the action tokens and change on every step. All 24 blocks re-run over
all 262 tokens, every step. That is the latency floor, and `num_denoising_steps` is the
only real lever on it.

**The loop is not Euler integration.** It re-forms `x_t` by interpolating between a
fixed noise draw and the current action estimate, and the transformer predicts the clean
action directly:

```python
x1 = randn(...); action = zeros_like(x1)
for i in range(steps, 0, -1):
    t = i / steps
    x_t = x1 * t + action * (1 - t)
    action = transformer(action_with_noise=x_t, t=t, ...)
```

Porting SmolVLA's `x_t += dt * v_t` here produces plausible-looking garbage — actions in
the right range, smooth, and wrong. Check parity against the PyTorch reference before
trusting any output.

What *is* hoisted out of the loop — the conditioning projections, their positional
embedding slice, the soft prompts — is exact, not an approximation: none of it depends
on `x_t` or `t`.

## Export parameters

| | |
|---|---|
| source | `lerobot/xvla-base` |
| `--valid-views` | 1 (single camera) |
| `--lang-len` | 50 |
| `--domain-id` | 0 |
| per-engine budget | 0.40 GB FP32 |
| opset | 17 |

`valid_views` is baked in, and it is worth being precise about what it changes. Exporting
the same checkpoint at `--valid-views 1` and `--valid-views 2` and diffing the result:

| graphs | 1 view vs 2 views |
|---|---|
| `vision_0..3` | **differ** — input batch dimension |
| `text_encoder_0..2`, `cond`, `denoise_0..3` | **byte-identical** |

`seq_len` is 262 in both. The sequence is sized by `num_image_views` (3, from the
checkpoint), not by how many views are real, so the token slots are paid for whether or
not a camera fills them.

**A second camera costs exactly one more pass through the vision tower** — it does not
lengthen the sequence and does not touch the denoise stack, which is the part that runs
once per denoising step. Re-export only to change the vision batch; nothing else moves.

## Running it

A reference runtime (ONNX Runtime + TensorRT EP, denoising loop in numpy) and a
benchmark harness live in
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla):

```bash
python -m bench ort-split --model xvla-base --bundle <this repo> --precision fp16
```

Engines are hardware- and version-specific: build them on the target board, one
subprocess per graph. Two builds in one process is enough to OOM 8 GB.

## Provenance

Exported with `tools/export_split_onnx.py` from the X-VLA runtime work in
[spark-projects](https://github.com/eetmie/spark-projects), under lerobot 0.6.1 /
transformers 5.5.4 / torch 2.11, opset 17.

Parity against the PyTorch reference on identical seeded observations and an identical
`x1` draw, both driven through the same preprocessing:

| | |
|---|---|
| cosine (min over samples) | **1.0000000** |
| max abs diff | 6.56e-07 |
| mean abs diff | 9.52e-08 |
| chunk shape | (30, 20) |

That is machine precision — the split is a re-serialization, not an approximation.

Measured on an Orin Nano 8 GB (JetPack 7.2, FP16 TRT, one camera, 10 denoising steps):
390 ms per 30-action chunk, 2.56 Hz replan, 5.71 GB peak RSS.
