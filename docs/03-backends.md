# 3. What is being compared

Two model families × four runtimes, on one board, with the same seeded observations.

## Models

```bash
python -m bench models          # the registry, with defaults and caveats
```

| key | family | params | PyTorch | split ONNX |
|---|---|---|---|---|
| `smolvla-base` | smolvla | 450 M | `lerobot/smolvla_base` | `ainekko/smolvla_base_onnx` |
| `xvla-base` | xvla | 880 M | `lerobot/xvla-base` | none published — export it |

Both are Apache 2.0. Anything not in the registry is benchmarked by pointing
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
8 GB. Budget ~5 min for SmolVLA, ~10 for X-VLA; afterwards they load from cache.

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

### `tether` — an off-the-shelf deployment CLI

[FastCrest Tether](https://github.com/FastCrest/tether) exports, verifies and serves VLA
policies behind an HTTP endpoint. It is included because it is a ready-made alternative
to hand-rolling an export, it supports both families here, and it is the kind of thing
someone will reasonably ask about. No affiliation, no endorsement — it is one row in the
table.

Its hardware table lists SmolVLA on an 8 GB Orin Nano at ~25 ms FP16, which is well
below what the split TensorRT path measures for a full denoise loop on this board
(vision 33 ms, prefill 16.5 ms, decode 11.4 ms × steps). That could mean several things
— a single forward pass rather than the whole loop, adaptive step early-exit, a
different board state — and the benchmark simply reports what happens here rather than
adjudicating. If it does not build or does not serve, the run is recorded as a failure
and the comparison moves on; **this backend is optional and `run_all.sh` skips it
unless `TETHER_EXPORT` is set.**

Two things it measures around. **It is a server** — the model is in another process, so
CPU/RSS attribution follows that PID. **The timing includes transport** — `total` is the
client roundtrip (PNG encode, JSON, loopback HTTP); where the response carries a
server-side figure it is reported separately as `server`. Compare `server` against the
other backends' `total` for a model-to-model number, and the roundtrip when asking what
a control loop would see. See `docs/04-metrics.md`.

The `/act` schema is not pinned in Tether's public docs, so the first call negotiates
across several payload shapes and records the accepted one. If none work:

```bash
python -m bench tether-probe --url http://127.0.0.1:8000   # its own OpenAPI schema
python -m bench tether --payload-template my_payload.json ...
```

Its fast path is a monolithic ONNX, which is the graph shape that will not TRT-build
here — so run `ort-mono` first. That separates "this runtime is slow" from "the monolith
cannot build on this board", which is a distinction worth having before drawing any
conclusion about anyone's tooling.

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

**The lerobot version matters, in an unobvious way.** The exporter reaches into
lerobot's *vendored* Florence2 (`lerobot/policies/xvla/modeling_florence2.py`), and that
module's layout has changed between releases: some revisions expose
`vlm.multi_modal_projector` and DaViT conv/block modules taking a bare tensor, others
fold the projector into `image_projection` + `image_proj_norm` and pass `(x, input_size)`
pairs through the tower. A mismatch fails with an `AttributeError` or a tracing error
that names neither the real cause nor the fix. Export in a venv pinned to the version
the exporter was written against (0.6.1); `scripts/export_xvla_split.sh` checks this
before it starts.

Both base checkpoints are Apache 2.0, so a derived ONNX export is redistributable with
attribution. Publishing the X-VLA split export to the Hub would make this comparison
reproducible by anyone with the same board — right now the SmolVLA half is (thanks to
`ainekko/smolvla_base_onnx`) and the X-VLA half is not. `hf/xvla-base-onnx-README.md` is
a model card written to go with such an upload.
