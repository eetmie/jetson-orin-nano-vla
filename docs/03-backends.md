# 3. Model profiles and runtimes

The repository has a PyTorch reference path and split-ONNX deployment paths:

| backend | purpose |
|---|---|
| `torch` | run a public upstream LeRobot base checkpoint |
| `ort-split` | run a matching split ONNX bundle through TensorRT EP |
| `ort-split` + `evo1-bootstrap` | validate and measure the nondeployable EVO1 export |
| `trt-split` | run a GR00T N1.6 or N1.7 split bundle on the TensorRT runtime alone |

Five profiles are registered. The EVO1 profile is deliberately not fetchable or
deployable because its current action head is random.

| key | parameters | Torch checkpoint | split ONNX |
|---|---:|---|---|
| `smolvla-base` | 450M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | [`eetmie/smolvla-base-onnx`](https://huggingface.co/eetmie/smolvla-base-onnx) |
| `xvla-base` | 880M | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | [`eetmie/xvla-base-onnx`](https://huggingface.co/eetmie/xvla-base-onnx) |
| `groot-n16-base` | 3.3B (2.3B deployed) | [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B), stock-PyTorch fixture in bundle | local export (`export/export.sh`) |
| `groot-n17-base` | 3.1B (2.5B deployed) | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B), stock-PyTorch fixture in bundle | local export (`export/export.sh`) |
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

## GR00T N1.6 base contract

`export/export.sh nvidia/GR00T-N1.6-3B <out>` writes 26 graphs (vision ×5, LLM ×8,
`cond`, `time`, DiT ×11), each about 105 M params, the size this board builds with
margin. Its fixed contract:

- embodiment `robocasa_panda_omron` (3 cameras), sliced from the 32-embodiment tables;
- 252×252 RGB per view, 81 image tokens each, through the stock eval transform;
- one prompt baked as token ids, right-padded to 320 (the LLM is causal and every DiT
  cross-attention masks the pad, so padding is exact);
- a 50-action chunk, 128-wide padded state and action, 4 Euler steps;
- mixed FP16 with LayerNorm, Softmax, every Qwen3 RMSNorm and the time sinusoids FP32.

It runs on `trt-split`, not ORT. ORT's TensorRT EP keeps every ONNX initializer in host
memory beside the engines, and on unified memory both count: about 5.5 bytes/param for
X-VLA here, which would be ~13 GB for GR00T's 2.3 B deployed params. The engines alone
hold each weight once, and all 26 contexts share one scratch buffer. The 621 MB token
embedding is a memory-mapped `.npy` gathered on the CPU.

The bundle carries stock-PyTorch FP32 outputs for seeded inputs (`fixture.npz`). Load
runs the engines on them and fails closed below cosine 0.999 or above 1 % of range on
the full chunk. There is no PyTorch run on the board: the BF16 checkpoint alone is
6.6 GB.

```bash
.venv-ort/bin/python -m bench trt-split --model groot-n16-base \
    --bundle ~/bundles/groot-n16-base-split --iters 100
```

## GR00T N1.7 base contract

`export/export.sh nvidia/GR00T-N1.7-3B <out>` writes 26 graphs: vision ×4 (the Qwen3-VL
ViT of Cosmos-Reason2-2B; the three DeepStack mergers ride in their layer's chunk), LLM
×8, `cond` ×2 (vlln, the 4-layer VL self-attention and the state encoder), `time`, DiT
×11. Its fixed contract:

- embodiment `xdof_relative_eef_relative_joint` (3 cameras), the 3-camera pretrained
  N1.7 embodiment; robocasa is not one;
- each camera seen twice, now and 30 frames earlier (the checkpoint's
  `video_delta_indices`); 256×256 per image through the stock eval transform, which
  already lands on a multiple of 32, so Qwen3-VL's own resize is a no-op;
- 64 tokens per image, 6 images; one prompt baked as token ids, 412 tokens right-padded
  to 448;
- a 40-action chunk, 132-wide padded state and action, 4 Euler steps;
- mixed FP16 with LayerNorm, Softmax and every RMSNorm FP32, the time sinusoids FP32.

What the split reproduces from stock, each checked against the PyTorch model: the
backbone output is the last decoder layer *before* the final RMSNorm (a ~1.5e4 activation
reaches vlln, which stays FP32); ViT layers 5/11/17 add DeepStack features to the
hidden state after LLM layers 0/1/2; the VL self-attention gets a key bias that hides the
padding, which stock never has.

Each camera frame's vision output depends on that frame alone, so `trt-split` encodes
the 3 current frames per call and reuses the earlier encode for the history slot: the
newest frame at least 1 s old at an assumed 30 fps camera (`meta.history_age_ms` records
what was used). The first call uses the current frame for both slots.

```bash
.venv-ort/bin/python -m bench trt-split --model groot-n17-base \
    --bundle ~/bundles/groot-n17-base-split --iters 100
```

Engines go to `~/.cache/jetson-orin-nano-vla/groot-n17-base-trt` (N1.6 keeps
`groot-trt`): graph names repeat between the two bundles.

## Why split ONNX

TensorRT temporarily materializes FP32 working copies while building an engine. A
whole-policy build exceeds the Orin Nano's 8 GB unified-memory budget. Building and
loading the split graphs one at a time keeps the peak within the board's budget.

Engine caches are tied to the exact JetPack, TensorRT, CUDA, GPU, graph, and precision.
Keep them on the Jetson and use a new cache after any of those inputs changes.
