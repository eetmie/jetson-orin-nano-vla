# 3. Model profiles and runtimes

The repository has a PyTorch reference path and split-ONNX deployment paths:

| backend | purpose |
|---|---|
| `torch` | run a public upstream LeRobot base checkpoint |
| `ort-split` | run a matching split ONNX bundle through TensorRT EP |
| `ort-split` + `evo1-bootstrap` | validate and measure the nondeployable EVO1 export |
| `trt-split` | run a split bundle from `export/export.sh` on the TensorRT runtime alone (every family) |

Seven profiles are registered (`python -m bench models`). `evo-depth-libero` is a
placeholder with no LeRobot-format weights; `evo1-bootstrap` is not deployable because its
action head is random.

| key | parameters | Torch checkpoint | split ONNX |
|---|---:|---|---|
| `smolvla-base` | 450M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | `export/export.sh`, or [`eetmie/smolvla-base-onnx`](https://huggingface.co/eetmie/smolvla-base-onnx) for `ort-split` |
| `xvla-base` | 880M | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | `export/export.sh`, or [`eetmie/xvla-base-onnx`](https://huggingface.co/eetmie/xvla-base-onnx) for `ort-split` |
| `evo1-libero` | 775M | [`zuoxingdong/evo1_libero`](https://huggingface.co/zuoxingdong/evo1_libero), native fixture in bundle | `export/export.sh` |
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

## Pure TensorRT (`trt-split`)

Every family runs on the TensorRT runtime alone when its bundle comes from
`export/export.sh`. ORT's TensorRT EP keeps the ONNX initializers in host memory beside
the engines, and on unified memory both count; the engines alone hold each weight once,
and all contexts share one scratch buffer. Against the retained ORT runs, system RAM in
use fell from 5.41 to 2.82 GB for X-VLA, 6.00 to 2.50 GB for EVO1 LIBERO and 2.45 to
1.74 GB for SmolVLA, at about the same latency.

An exported bundle carries what the runtime needs and ORT does not:

- `fixture.npz` (EVO1: `parity_fixture.npz`): the stock LeRobot policy's FP32 output for
  seeded inputs and noise. Load runs the engines on it and fails closed below cosine
  0.999 or above 1 % of range on the full chunk;
- mixed-FP16 graphs whose precision the strongly typed build follows as written:
  LayerNorm, Softmax and every RMSNorm stay FP32 (`export/vla_common/fp16_mixed.py`);
- SmolVLA and EVO1: the token embedding as `embed_tokens.npy`, memory-mapped on the CPU,
  instead of running the Gather-only text graph.

Engines are built one subprocess at a time into `~/.cache/jetson-orin-nano-vla/<model>-trt`
and keyed by the ONNX sha256, the TensorRT and CUDA versions and the builder options.

`--chain` picks how the engines are driven:

| chain | what runs between engines |
|---|---|
| `graph` (default) | nothing on the host: the whole chain stays on the GPU and is replayed as a captured CUDA graph (GR00T N1.7: two graphs around its frame history) |
| `device` | the same device-resident chain, enqueued call by call |
| `host` | numpy: every output comes back to the CPU and the next input goes up again |

The glue the host chain does in numpy runs on the GPU instead: device-to-device copies
for the image-token scatter, and tiny FP32 TensorRT op engines (`bench/vendor/trt_ops.py`)
for X-VLA's interpolation, the Euler updates and SmolVLA's SiLU. Inputs depending only
on the schedule or the prompt (timestep embeddings, masks, prompt rows) are computed once.
At load, `graph` and `device` must reproduce the `host` chain and the bundle fixture on
the fixture's inputs; `meta.fixture_parity.device_chain` records both comparisons.

```bash
.venv-ort/bin/python -m bench trt-split --model xvla-base \
    --bundle ~/bundles/xvla-base-split --iters 100
```

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

## EVO1 LIBERO contract

`export/export.sh zuoxingdong/evo1_libero <out>` exports the trained LIBERO checkpoint
with the same eleven-graph layout as the bootstrap below, at its two cameras: a
576-token sequence (2 × 256 image tokens + 64 text), 8-wide state and 7-wide action padded
to 24, 32 Euler steps. Its actions mean something for LIBERO's embodiment only.

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
`cond`, `time`, `mod` ×2, `kv`, DiT ×8), each about 105 M params, the size this board builds with
margin. Its fixed contract:

- embodiment `robocasa_panda_omron` (3 cameras), sliced from the 32-embodiment tables;
- 252×252 RGB per view, 81 image tokens each, through the stock eval transform;
- one prompt baked as token ids, right-padded to 320 (the LLM is causal and every DiT
  cross-attention masks the pad, so padding is exact);
- a 50-action chunk, 128-wide padded state and action, 4 Euler steps;
- mixed FP16 with LayerNorm, Softmax, every Qwen3 RMSNorm and the time sinusoids FP32.

Two DiT inputs never change within an inference, so they leave the per-step graphs (the
idea comes from FlashRT's π0.5 runtime). Each block's AdaLN modulation depends only on
the timestep: the `mod` graphs compute it for the fixed schedule at load and are then
unloaded. Each cross-attention block's keys and values depend only on the backbone
features: the `kv` graph computes them once per observation instead of once per step.
The DiT blocks are fed both through stand-ins for their norm and K/V projections, so the
stock block code runs unchanged.

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
×8, `cond` ×2 (vlln, the 4-layer VL self-attention and the state encoder), `time`,
`mod` ×2, `kv` and DiT ×8, split the same way as N1.6's. Its fixed contract:

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

## pi0.5 LIBERO (prototype)

`trt-split --model pi05-libero` runs Physical Intelligence's `pi05_libero` from a bundle
made by the scripts in `export/pi05/` (not yet `export/export.sh`). FP16 weights and
hidden states, FP32 normalization, accumulation and Euler updates; 10 steps, horizon 10.
Each layer kind (SigLIP, Gemma language, action expert and four small ones) is one
weight-stripped, refittable template built on the board; all 67 components are copies
refitted with their own weights at load, so a build needs one layer's memory. TensorRT
allocates through plain `cudaMalloc` (its default allocator failed mid-load on this
board). The AdaRMS modulation is a fixed table for the 10-step schedule; prompt rows come
from a memory-mapped FP16 embedding table; the prompt is tokenized at export (`tasks`).

Two layouts: padded (968-token prefix: three image slots, the third masked, 200 prompt
slots) and compact (521: the two real cameras and the prompt's real tokens, the masked
camera never computed). Load checks every bundle fixture against its full-FP16 PyTorch
trajectory: cosine >= 0.99999 and max |diff| <= 0.005 on the normalized chunk.

## Why split ONNX

TensorRT temporarily materializes FP32 working copies while building an engine. A
whole-policy build exceeds the Orin Nano's 8 GB unified-memory budget. Building and
loading the split graphs one at a time keeps the peak within the board's budget.

Engine caches are tied to the exact JetPack, TensorRT, CUDA, GPU, graph, and precision.
Keep them on the Jetson. `trt-split` rebuilds an engine whose ONNX, TensorRT/CUDA version
or builder options changed; for `ort-split`, use a new cache after any of those changes.
