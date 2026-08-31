# 3. Base models and runtimes

The repository has one reference path and one deployment path:

| backend | purpose |
|---|---|
| `torch` | run the upstream LeRobot base checkpoint |
| `ort-split` | run the matching split ONNX bundle through TensorRT EP |

Only `smolvla-base` and `xvla-base` are registered.

| key | parameters | Torch checkpoint | split ONNX |
|---|---:|---|---|
| `smolvla-base` | 450M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | [`eetmie/smolvla-base-onnx`](https://huggingface.co/eetmie/smolvla-base-onnx) |
| `xvla-base` | 880M | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | [`eetmie/xvla-base-onnx`](https://huggingface.co/eetmie/xvla-base-onnx) |

The split repositories are private until publication. Authenticate once with
`hf auth login` while they remain private.

## Download

```bash
scripts/fetch_models.sh smolvla-base
scripts/fetch_models.sh xvla-base
```

Each command writes the upstream checkpoint to `~/bundles/<model>-torch` and the
matching ONNX graphs to `~/bundles/<model>-split`.

## SmolVLA base contract

The verified bundle contains nine graphs: vision, text, expert prefill, expert decode,
state, action-input, action-output, time-input, and time-output. Its fixed contract is:

- two 512×512 camera slots;
- a 177-token prefix with 48 language tokens;
- a 50-action chunk;
- ten denoising steps;
- 32-wide padded state and action tensors.

The runtime sends the large graphs to TensorRT and uses IOBinding for the cached
denoising state. The tokenizer, normalization statistics, `export_info.json`, and all
graph files are part of one bundle and must stay together.

```bash
.venv-ort/bin/python -m bench ort-split \
    --model smolvla-base \
    --bundle ~/bundles/smolvla-base-split \
    --views 2 --iters 100
```

## X-VLA base contract

The verified bundle contains twelve graphs: four vision stages, three text stages, one
conditioning graph, and four denoiser stages. Its fixed contract is:

- three 224×224 image views with 50 tokens per view;
- a 50-token language sequence;
- a 30-action chunk;
- ten denoising steps;
- 20-wide state and action tensors in `ee6d` mode.

X-VLA has no prefill/decode KV-cache seam: its bidirectional policy transformer reruns
on every denoising step.

```bash
.venv-ort/bin/python -m bench ort-split \
    --model xvla-base \
    --bundle ~/bundles/xvla-base-split \
    --views 3 --iters 100
```

## Why split ONNX

TensorRT temporarily materializes FP32 working copies while building an engine. A
whole-policy build exceeds the Orin Nano's 8 GB unified-memory budget. Building and
loading the split graphs one at a time keeps the peak within the board's budget.

Engine caches are tied to the exact JetPack, TensorRT, CUDA, GPU, graph, and precision.
Keep them on the Jetson and use a new cache after any of those inputs changes.
