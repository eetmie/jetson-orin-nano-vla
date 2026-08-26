# 3. What is being compared

Two model families × three runtimes, on one board, with the same seeded observations.

## Models

```bash
python -m bench models          # the registry, with defaults and caveats
```

| key | family | params | PyTorch | split ONNX |
|---|---|---|---|---|
| `smolvla-base` | smolvla | 450 M | `lerobot/smolvla_base` | `ainekko/smolvla_base_onnx` |
| `xvla-base` | xvla | 880 M | `lerobot/xvla-base` | none published — export it |

Both checkpoints are Apache 2.0 and stay under that licence; this repository's own code
is MIT. Anything not in the registry is benchmarked by pointing
`--checkpoint` / `--bundle` at it and passing `--family`.

**Base weights produce meaningless actions.** They were never fine-tuned on a robot
here. That is fine for latency, memory, CPU and power, which is what the base entries
are for, and it is why parity is always measured against another runtime of the *same*
weights rather than against task success.

### The two families are genuinely different problems

| | SmolVLA 450 M | X-VLA 880 M |
|---|---|---|
| conditioning | vision + text + expert prefill → KV cache | DaViT + BART → conditioning tokens |
| hot loop | expert **decode** ×N over the cached KV | policy transformer, **all 24 blocks, all 262 tokens**, ×N |
| KV cache in the loop | yes | **impossible** |
| loop maths | Euler: `x_t += dt · v_t` | interpolate a fixed noise draw against the current estimate; the transformer predicts the clean action |
| split engines needed | 9 | 12 |

Two things follow that shape the whole repo. X-VLA has **no prefill/decode seam** — its
policy transformer is a bidirectional encoder, so the conditioning tokens attend *to*
the action tokens and change on every denoising step. Every block re-runs, every step.
The only real lever on its latency is the step count. And its loop is **not**
integration: porting SmolVLA's update produces plausible-looking garbage, which is
precisely the failure mode worth naming out loud.

Prior measurements on this board, one real camera, 10 steps, FP16 TRT:

| | SmolVLA 450 M | X-VLA 880 M |
|---|---:|---:|
| engines | 9 | 12 |
| chunk latency | 210 ms | 390 ms |
| actions per chunk | 50 | 30 |
| replan rate | 4.8 Hz | 2.56 Hz |
| **peak RSS** | **2.21 GB** | **5.71 GB** |
| free RAM left of 7.4 GB | 4.86 GB | 1.47 GB |
| bytes per parameter resident | ~4.4 | ~6.9 |

X-VLA is 2× the parameters but **2.6× the resident memory**, and cost per parameter is
worse — which points at TensorRT per-engine activation memory rather than weights. That
is the difference between 4.9 GB of headroom for the rest of the robot and 1.5 GB.

## How many cameras — and what the second one actually costs

NVIDIA's guidance for the Orin Nano is at most two USB cameras, and that is a real
constraint on a real deployment. It is a **capture-side** constraint, though: the
benchmark feeds observations from memory, so it measures what a second camera costs the
*model*, not what it costs the USB controller. Both numbers matter and they are
different numbers.

What the model side costs is now measured rather than assumed. The X-VLA export was
produced twice, at `--valid-views 1` and `--valid-views 2`, and the two bundles were
compared file by file:

| graphs | 1 view vs 2 views |
|---|---|
| `vision_0..3` | **differ** (input batch dimension) |
| `text_encoder_0..2`, `cond`, `denoise_0..3` | **byte-identical** |

`seq_len` is 262 in both. The sequence is sized by `num_image_views` (3, declared by the
checkpoint), not by how many of those views are real — so the token slots are paid for
whether or not a camera fills them.

**A second camera therefore costs exactly one more pass through the vision tower.** It
does not lengthen the sequence, and it does not touch the denoise stack, which is the
part that runs ten times per chunk. On the SmolVLA side the same thing holds by
construction: the padded slot's embedding is computed once at load and reused for the
life of the process, so an empty slot costs zero vision passes per inference while still
occupying its 64 tokens in the prefix.

That has a practical consequence worth stating plainly: **if your export already carries
two slots, running one camera does not make inference much cheaper.** You have already
paid for the sequence. The cheap configuration is a *one-slot export*, not a two-slot
export fed one image.

### So: sweep 1 and 2, do not collapse to 2

Two cameras is the right deployment ceiling and belongs in the headline table. But 1-vs-2
is the measurement that prices the second camera, and it is one flag:

```bash
python -m bench ort-split --model smolvla-base --bundle $BUNDLE --views 1 --label sv.1cam
python -m bench ort-split --model smolvla-base --bundle $BUNDLE --views 2 --label sv.2cam
```

Three configurations are worth distinguishing, and the run metadata records which is
which (`views`, `cam_slots`):

| config | vision passes | prefix / sequence |
|---|---|---|
| 1-slot export, 1 camera | 1 | short |
| 2-slot export, 1 camera | 1 | long (slot padded) |
| 2-slot export, 2 cameras | 2 | long |

Confirmed on the public two-slot export (`ainekko/smolvla_base_onnx`), CPU providers, so
read the call counts and the flat rows rather than the absolute times:

| | vision | vision calls | prefill | decode |
|---|---:|---:|---:|---:|
| `--views 1` | 944 ms | **1** | 418 ms | 1648 ms |
| `--views 2` | 1600 ms | **2** | 356 ms | 1529 ms |

Vision scales with the camera count; prefill and decode do not move. Same conclusion the
X-VLA bundle diff reached structurally, now reached empirically on the other family.

`--views` sets the real cameras; `--cam-slots` sets what the export was built with.
PyTorch pads to `cam_slots` with lerobot's all -1 convention behind a False mask,
because otherwise it would build a 113-token prefix against the ONNX path's 177 and the
structural mismatch would surface as a parity failure that looks like a numerics bug.

### What this does NOT measure, by design

**Capture, and anything else touching real hardware.** This repo answers "does the model
fit, and what does it cost" — it does not drive a robot, open a camera, or measure a
control loop. Observations come from memory precisely so that a latency number is a
property of the runtime and not of a camera's timing.

So the numbers here exclude: USB bandwidth, frame decode, video encode, and the CPU all
three take. On a board where CPU headroom is the point, that is a real omission and it
belongs in any writeup — but it is a separate measurement on a separate rig, not a gap
to be plugged here.

Two figures worth carrying across when someone does read these against real hardware. A
RealSense D435i delivers IR and colour from **one** USB device on one pipeline, so an
IR + RGB pair is one device, not two, and the two-camera guidance binds less tightly
than it sounds. And measured on that setup at 640×480×30 for both streams: 0 dropped
frames, USB ~6% — the expensive part was the second video *encode*, which is host CPU
rather than bandwidth.

## Backends

### `torch` — stock PyTorch

The control baseline. Load the checkpoint the way the training repo does and call the
model. No export, no engine, no build step. Everything the other backends buy is paid
for against this.

For SmolVLA there are three precision variants, because "FP16 PyTorch" is ambiguous and
the difference matters on an 8 GB board:

| flags | what it is |
|---|---|
| `--weights float32 --autocast off` | the numerical reference |
| `--weights float32 --autocast float16` | mixed precision as LeRobot runs it (`use_amp: true`) — FP16 matmuls, FP32 master weights |
| `--weights float16 --patch-half-out` | hard cast; halves resident weights |

The hard cast needs a patch. LeRobot 0.5.1 hardcodes
`suffix_out.to(dtype=torch.float32)` in `denoise_step`, so a half-weights policy dies
with *"mat1 and mat2 must have the same dtype"*. `--patch-half-out` wraps that one
projection to cast to its own weight dtype; it is a deliberate deviation from stock
LeRobot and is recorded as `patched_half_out: true`.

X-VLA's `generate_actions` draws its own `x1` internally rather than accepting a
`noise=` argument, so the backend swaps `torch.randn` for the duration of the call and
intercepts only the exact target shape. Without that there is no element-wise parity
against the split export at all — only a distribution comparison.

### `ort-split` — the split ONNX export on ORT + TensorRT EP

The deploy path for both families, and it is split because the monolith **cannot be
built on this board**. That is measured: the 108k-node 10-step SmolVLA graph OOM'd or
errored on every combination of FP16/FP32, GUI/headless and builder knobs, taking up to
85 minutes to fail. Cutting to 5 steps (61k nodes) barely moved the peak — 6.7 GB
against 7.4 GB — because the floor is node-count independent: TensorRT imports all the
weights as FP32 working copies before it optimizes anything.

For X-VLA the same constraint was turned into a formula on this board:

```
build peak RSS  ≈  3.18 GB  +  5.63 × (FP32 weight GB)
```

which leaves room for about 0.4 GB of weights (~100 M params) per engine. All three of
X-VLA's heavy components exceed that alone, hence twelve engines.

**Where the time goes is instrumented, not assumed.** Every ORT session is wrapped in a
timer, so each run reports per-graph totals and call counts, plus `graphs_gpu`,
`graphs_cpu` and `python_numpy`. For SmolVLA it also reports `per_step_projectors` and
`decode_trt` separately, because the stock runtime puts the text encoder and all five
projectors on the **CPU** execution provider — and `action_in`, `time_in`, `time_out`
and `action_out` each run once per denoising step. "It runs on the GPU" is a claim
about three of nine graphs. `--projectors gpu` rebuilds those four on the TRT/CUDA
stack so the alternative can be measured rather than argued about.

First run builds every engine, one subprocess per graph. Two builds in one process OOM
8 GB. Budget around 5 min for SmolVLA's nine, longer for X-VLA's twelve (not yet
timed); afterwards they load from cache in seconds.

### `ort-mono` — the monolith A/B

The whole model as one graph, denoise loop unrolled, on the TensorRT EP or (with
`--no-trt`) the CUDA EP. Nobody publishes an honest number for this on an 8 GB Orin,
which is exactly why it is worth taking — it is the counterfactual that decides whether
the split is an optimization or a necessity. Prior: on the CUDA EP with no engine build
to fail, the SmolVLA monolith ran at **532 ms mean / 498 ms p50**, finite and plausible.

Three outcomes are distinguished, and the backend records which happened:

| outcome | meaning |
|---|---|
| TensorRT built | the split is an optimization, not a necessity |
| CUDA fallback | the ~500 ms regime; still viable for a chunked policy |
| CPU fallback | a wrong number waiting to be quoted as a right one |

That last case is why `active_provider`, `trt_engine_cached` and `ran_on_gpu` are in
every result. When TensorRT gives up, ONNX Runtime does not error — it partitions
elsewhere and keeps going, still returning a finite, plausible action chunk.

## The published SmolVLA ONNX does not match the checkpoint under lerobot 0.5.1

`ainekko/smolvla_base_onnx` is the only public split export of `lerobot/smolvla_base`,
and this repo used it as `smolvla-base`'s bundle. Measured against a PyTorch reference
running lerobot 0.5.1, on the same board, through the same runtime and harness, with
the same observation:

| bundle | cosine | max abs diff | % of range | first action |
|---|---:|---:|---:|---:|
| exported here (`--cam-slots 2`) | **0.9993839** | 0.3439 | **1.94%** | 0.088 |
| `ainekko/smolvla_base_onnx` | 0.9393259 | 2.2940 | **12.92%** | 0.657 |

12.9% of commanded range is not a numerical artefact. It is a difference you would feel
on a machine, and nothing errors — the actions are finite, plausible and wrong.

**It is not the weights.** All 198 initializers in ainekko's vision graph appear in
ours: 125 exactly, 73 under transpose (they store `W` and emit `Transpose -> MatMul`,
we pre-transpose into `MatMul`). None unmatched, `max|d| = 0.000e+00`.

**It is not precision or TensorRT.** The same graphs on the CPU EP at fp32 deviate
identically — 0.9543 against 0.9541 for the fp16 TRT run.

**It is not preprocessing, padding or the tokenizer.** Both backends share
`resize_with_pad_uint8` and `normalize_state`; the native run has zero padded slots and
still shows it; the text graph is *bit-identical* between the two exports
(`max|d| = 0.000e+00`).

**It is the vision graph.** Feeding one fixed input through both, stage by stage on the
CPU EP:

```
smolvlm_text     max|d| = 0.0000e+00   cos = 1.0000000
smolvlm_vision   max|d| = 2.8263e+01   cos = 0.8221545
```

Same weights, different wiring: 1419 nodes against our 530, LayerNorm decomposed into
primitives, and no input convention that reconciles them (`[-1,1]`, `[0,1]`,
ImageNet-normalized and `[0,255]` were all tried; `[-1,1]` is closest, so the runtime
feeds it correctly). That reads like an export traced from a different lerobot, whose
`embed_image` differs.

**Read this carefully before calling the published export wrong.** Both exports compared
here were traced under lerobot 0.5.1 and scored against a 0.5.1 PyTorch reference, so
ours is the same code traced twice — consistent, not independently canonical. Against
whatever lerobot ainekko used, theirs may well score 0.9999 and ours may be the outlier.
What is established is narrower and still worth knowing: **under lerobot 0.5.1, that
export produces actions 12.9% of range from the checkpoint it is named after.**

## `torch` for X-VLA does not currently run

`bench/backends/torch_xvla.py` has never produced a result row. It fails before the
first inference with

```
ValueError: Sequence length 1204 exceeds max_len_seq=512
```

raised inside X-VLA's own `soft_transformer.py`, not Florence2. The split ONNX path
builds a 262-token sequence for the same observation; the PyTorch path builds 1204, and
the length does not change with `--views`, so it is not the camera count. Image shapes
were checked and are correct — `(1, 3, 3, 224, 224)`, resized by `preprocess_image`.

Unresolved. It is a discrepancy between how this backend drives the policy and how the
export does, and it needs instrumented shapes rather than inference. Three environment
bugs were fixed on the way to reaching it, all of which suggest the path had simply
never been exercised: `13_env_torch_xvla.sh` installed torch 2.13.0 (whose
`libtorch_cuda.so` wants `ncclCommResume`, absent on JetPack 7.2) where 2.11.0+cu130 is
required, and a PyTorch-only run needs `onnxruntime` installed because the vendored
`xvla_split_ort` imports it at module scope just to provide `preprocess_image`.

X-VLA parity is therefore quoted from `xvla-runtime/parity.py` — cosine 0.999993
(action) and 0.999953 (cond_tokens) — which builds both sides' inputs in one script at
CPU fp32. That is a **less demanding test** than the cross-backend comparison used for
SmolVLA, and should be cited as such rather than presented as equivalent.

## Getting the artefacts

```bash
python -m bench fetch --model smolvla-base --what both     # torch + split ONNX
python -m bench fetch --model xvla-base    --what torch    # no split export published
```

`scripts/fetch_models.sh` does the same with the `hf` CLI, whose resume behaviour is
better: `snapshot_download` has been observed to stall on large safetensors here —
process alive, file not growing, no exception, so its own retry never fires. The script
documents a `curl --speed-limit/--speed-time -C -` fallback for that case.

### Exporting a split ONNX where none exists

X-VLA has no published ONNX. The exporter that produced the twelve-graph layout is
`spark-projects/orin-nano/xvla-runtime/tools/export_split_onnx.py`, wrapped here:

```bash
CHECKPOINT=~/models/xvla-base OUT=~/bundles/xvla-base-split scripts/export_xvla_split.sh
```

**Run it off the board.** Loading the policy costs ~3.5 GB on CPU and the exporter holds
that alongside an export trace; that fits on a workstation and swaps on 8 GB. Export
elsewhere, rsync the graphs over, build the engines on the target — a TensorRT engine is
hardware- and version-specific and is never copied. The ONNX is the portable artefact.

**The lerobot version matters, in an unobvious way.** Where Florence2 comes from changed
between releases. lerobot 0.6.1 uses **transformers'** `Florence2Model` — flat children
`vision_tower` / `multi_modal_projector` / `language_model`, DaViT modules taking a bare
tensor. Earlier revisions vendored their own `lerobot/policies/xvla/modeling_florence2.py`
with no `multi_modal_projector` at all (the projection is `image_projection` +
`image_proj_norm`) and conv/block modules threading `(x, input_size)` pairs through the
tower. Against the wrong one the export dies with an `AttributeError` or a tracing error
that names neither the cause nor the fix. Export in a venv pinned to 0.6.1;
`scripts/export_xvla_split.sh` checks the version before it starts.

Two smaller traps in the same exporter, both fixed upstream in the runtime project while
producing the export published here:

- The empty-input guard was over-broad. ONNX uses `""` as the legal way to omit a
  *trailing optional* input, which is exactly what the tracer emits for the DaViT
  window-attention `Pad` — `constant_value` omitted because at 224×224 every feature map
  divides evenly by the window size, so the pads are all zero. Flagging that rejected a
  graph `onnx.checker` passes and ORT loads. The guard now consults the op schema and
  only rejects an empty string in a *required* position, which is the failure it was
  written for.
- The text-encoder branch deletes `vision_tower` to free 1.4 GB before tracing, and then
  builds a module that looks the VLM up again. Any helper resolving those submodules has
  to tolerate the tower already being gone.

Both base checkpoints are Apache 2.0, so a derived ONNX export is redistributable with
attribution. Publishing the X-VLA split export to the Hub would make this comparison
reproducible by anyone with the same board — right now the SmolVLA half is (thanks to
`ainekko/smolvla_base_onnx`) and the X-VLA half is not. `hf/xvla-base-onnx-README.md` is
a model card written to go with such an upload.
