# 3. Model profiles and runtimes

The repository has a PyTorch reference path and split-ONNX deployment paths:

| backend | purpose |
|---|---|
| `torch` | run a public upstream LeRobot base checkpoint |
| `ort-split` | run a matching split ONNX bundle through TensorRT EP |
| `ort-split` + `evo1-bootstrap` | validate and measure the nondeployable EVO1 export |

Three profiles are registered. The EVO1 profile is deliberately not fetchable or
deployable because its current action head is random.

| key | parameters | Torch checkpoint | split ONNX |
|---|---:|---|---|
| `smolvla-base` | 450M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | [`eetmie/smolvla-base-onnx`](https://huggingface.co/eetmie/smolvla-base-onnx) |
| `xvla-base` | 880M | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | [`eetmie/xvla-base-onnx`](https://huggingface.co/eetmie/xvla-base-onnx) |
| `evo1-bootstrap` | 775M | native fixture in bundle | local checksummed export |

Authenticate once with `hf auth login` if a published split repository is private.

## Download the public bases

```bash
scripts/fetch_models.sh smolvla-base
scripts/fetch_models.sh xvla-base
```

Each command writes the upstream checkpoint to `~/bundles/<model>-torch` and the
matching ONNX graphs to `~/bundles/<model>-split`. `evo1-bootstrap` is not on this
download path: export it using the companion Spark workflow and copy the complete
bundle without copying TensorRT cache files.

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

## EVO1 bootstrap contract

The tested bundle contains eleven graphs: four vision stages, a CPU token embedding,
three language stages, action-context cache construction, the repeated action step,
and the action output. Its fixed contract is:

- one 448×448 RGB view represented by 256 image tokens;
- a 320-token language/vision sequence;
- a 50-action chunk with 24-wide state and action tensors;
- 32 Euler flow steps with injected uniform `[-1, 1]` noise;
- mixed FP16 weights with sensitive operations retained in FP32;
- `OpenGVLab/InternVL3-1B-hf` revision
  `014c0583a0d4bedf29fbe2dbff4f865eb998e171` as the pinned VLM initializer.

All eleven graphs, the tokenizer, `bundle.json`, and the native LeRobot 0.6.1 fixture
are covered by `MANIFEST.sha256`. The runtime accepts only schema 1, one-camera bundles
marked `deployable: false` and `random_action_head: true`. This prevents the bootstrap
from being mistaken for a trained policy; a trained RoboTwin or SO100 checkpoint needs
a distinct artifact and runtime contract.

The automatic fixture gate checks native graph-boundary tensors and the complete raw
RGB → resize/normalize → tokenize → eleven-graph action path. The retained FP16 run
reached cosine 0.999991 for the stored action and 0.999980 from the raw observation,
against a 0.999 threshold.

```bash
.venv-ort/bin/python scripts/check_evo1_fixture.py \
    --bundle ~/bundles/evo1-bootstrap-split \
    --cache-dir ~/.cache/jetson-orin-nano-vla/evo1-trt

.venv-ort/bin/python -m bench ort-split \
    --model evo1-bootstrap \
    --bundle ~/bundles/evo1-bootstrap-split \
    --cache-dir ~/.cache/jetson-orin-nano-vla/evo1-trt \
    --iters 100
```

Ten compute graphs use TensorRT; the large FP32 token embedding stays on CPU. The
measured fast path keeps the action-context K/V values and the intermediate action
hidden state on the GPU with IOBinding. CUDA EP fallback was removed after testing;
unsupported work falls back directly to CPU. The small Euler update remains on host.

Do not use `--num-steps` for the retained EVO1 result. Changing its 32-step native
contract changes the expected action and therefore fails the embedded action-parity
gate.

## Why split ONNX

TensorRT temporarily materializes FP32 working copies while building an engine. A
whole-policy build exceeds the Orin Nano's 8 GB unified-memory budget. Building and
loading the split graphs one at a time keeps the peak within the board's budget.

Engine caches are tied to the exact JetPack, TensorRT, CUDA, GPU, graph, and precision.
Keep them on the Jetson and use a new cache after any of those inputs changes.
