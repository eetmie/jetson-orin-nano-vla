# 6. Optimization backlog — ordered, and mostly not done yet

The split path works. This is the list of things that might make it faster, in the
order that keeps each result interpretable. **Measure each in isolation and keep the
numbers**; a stack of simultaneous changes tells you the total and nothing else.

Status legend: **done** = the harness already measures or supports it · **flag** =
supported, needs a run · **todo** = a real code change, not written yet.

---

## 0. Instrument first — **done**

> Before changing anything, wrap `sample_actions` with per-stage `perf_counter`:
> vision, prefill, and inside `_denoise_step` split out the four projector calls vs the
> TRT decode call. Everything below is a guess until you have that breakdown.

This is already in the harness and does not require touching the vendored runtime.
Every ORT session is wrapped in a delegating timer (`_TimedSession` in
`bench/backends/ort_split.py`), so each result JSON carries, per inference:

```
graph.vision            graph.vision.calls
graph.prefill           graph.prefill.calls
graph.decode            graph.decode.calls          -> 10 at the default step budget
graph.action_in/.time_in/.time_out/.action_out      -> 10 calls each
per_step_projectors     the four above, summed      <- the number in question
decode_trt              the TensorRT decode, alone
graphs_gpu / graphs_cpu / python_numpy
```

Wrapping rather than editing keeps the measured code identical to what runs on the
robot. `python -m bench report` prints the breakdown as its own table.

One data point already, from the CPU-provider plumbing test on the DGX Spark (so the
absolute values mean nothing, the ratio is the point): the four per-step projectors
summed to **16.5 ms** against **611 ms** of decode. If that ratio roughly holds on the
Orin, where decode is ~11 ms/step, the projectors are in the same order of magnitude as
your 30–60 ms suspicion rather than a rounding error — but that is a guess until it is
measured on the board, which is the entire point of item 0.

**Run:** any `ort-split` run produces this. No extra flag.

---

## 1. Move the projectors to GPU — **flag**

> Change `time_in` / `time_out` (and `action_in` / `action_out`) from cpu to heavy.
> Adds four more TRT builds. Fall back to `CUDAExecutionProvider` if TRT won't build them.

```bash
python -m bench ort-split --bundle $BUNDLE --projectors gpu --label ort-split.proj-gpu
```

Implemented by rebuilding those four sessions on the heavy provider stack *after* the
policy is constructed, so the vendored runtime stays byte-faithful. If TensorRT
declines a graph the EP stack falls through to CUDA by itself;
`configured_provider_priority_per_graph` records the intended first provider, and
`projector_move_failures` records any that refused outright.

A/B it against the same run with `--projectors cpu`. Watch the build peak on the first
run — they are tiny graphs and may not need the subprocess isolation, but "may not" is
not "does not" on this board.

---

## 2. Raise the TensorRT build settings — **flag**

> Try `TRT_OPT_LEVEL=3` (TRT 10's default) then 5, and `TRT_WORKSPACE_MB=1024` or 2048.
> Clear the cache between attempts or nothing rebuilds.

```bash
rm -rf ~/.cache/jetson-orin-nano-vla/trt          # or nothing rebuilds
python -m bench ort-split --bundle $BUNDLE --trt-opt-level 3 --trt-workspace-mb 1024 \
    --label ort-split.opt3-ws1024
```

The stock settings here are level 2 / 512 MB, chosen because they *fit*, not because
they are fast. Higher levels explore more tactics: a larger one-time build peak and a
longer build, for a possibly faster cached engine. One-time cost, cached after — free
speed if it fits. Both values are recorded in the run metadata, so an A/B cannot be
mixed up after the fact.

The cache path is deliberately `~/.cache/jetson-orin-nano-vla/trt`, not `/tmp`: `/tmp`
clears at boot and a cold build is ~5 minutes.

---

## 3. IOBinding for the denoise loop — **todo**

> Bind the 32 KV tensors once as device OrtValues after prefill instead of re-feeding
> numpy each step, and keep `x_t` on device between iterations.

The biggest refactor of the set, which is why it is behind 1 and 2 — those tell you how
much of the loop is copy overhead in the first place. Every step currently hands ORT 32
KV tensors plus masks and position ids as numpy, and takes `x_t` back to the host to do
`x_t += dt * v_t`.

Note the interaction with item 1: if the projectors move to the GPU, the per-step host
round trip is what remains, and IOBinding is what removes it. If they stay on the CPU,
IOBinding cannot help the projector calls at all. Do 1 first.

Requires editing the runtime rather than wrapping it. When it happens, do it in
`kaivuriprokkis` and re-vendor, so the robot and the benchmark stay the same code.

---

## 4. Precompute the time-embedding term — **todo**

> `time_in` is linear over `concat[action_emb, time_emb]`, so it splits into
> `W_a @ action_emb + (W_t @ time_emb + b)`. With `num_steps=10`, `t` takes ten known
> values — compute the second term once at load.

Exact, not an approximation, and it halves `time_in`'s per-step work. Also strictly
smaller than item 1: if the projectors are on the GPU already, this saves less.

**Only worth doing if item 0 shows `time_in` is significant.** The per-graph breakdown
names it directly (`graph.time_in`), so that decision is one run away.

---

## 5. Cut the denoise steps — **flag, and the one that can change behaviour**

```bash
python -m bench ort-split --bundle $BUNDLE --num-steps 5 --label ort-split.s5
python -m bench ort-split --bundle $BUNDLE --num-steps 4 --label ort-split.s4
```

Prior on the Spark: 10 → 5 steps was **~33% faster** with fidelity-vs-own-FP32 slightly
*better* (0.9985 vs 0.9974). The cost is coarser denoising, which is a task-quality
question, not a numerics one.

Everything else on this list changes only latency. This one changes what the policy
does, so it needs the extra checks:

- `python -m bench parity` against the same bundle at the full step count — necessary,
  not sufficient.
- the action↔motion correlation check on a recorded episode: healthy is r = 0.75..0.96
  at a 67–200 ms lag.
- eyeball a few episodes.

For X-VLA this is the *only* real latency lever, because there is no KV cache to
exploit and every block re-runs on every step.

---

## 6. The monolithic A/B — **flag**

> Export `lerobot/smolvla_base` monolithic on the Spark, rsync to the Orin, try to
> serve it. Log whether ORT-TRT builds it, OOMs, or silently drops to the CUDA EP —
> check `sess.get_providers()[0]` and don't trust it to be TRT.

Implemented as its own backend:

```bash
python -m bench ort-mono --onnx exports/smolvla_base_fp32_static.onnx --bundle $BUNDLE \
    --label mono.trt
python -m bench ort-mono --onnx exports/smolvla_base_fp32_static.onnx --bundle $BUNDLE \
    --no-trt --label mono.cuda-ep
```

It checks providers explicitly rather than trusting them, and every result carries
`active_provider`, `active_providers`, `trt_engine_cached`, `n_cached_engines` and
`ran_on_gpu`. A silent CPU fallback still returns a finite, plausible action chunk —
that is exactly the trap, and it is why these fields exist.



---

## Why this is worth writing down

A per-graph breakdown of a split VLA on an 8 GB Orin, plus a monolithic build result on
the same board, is not published anywhere. Keep the numbers per change, keep the
failures, and the backlog turns into a result rather than a to-do list.

---

# MEASURED — Orin Nano Super, JetPack 7.2, 2026-08-25

Board at MAXN_SUPER, stock 2 GB swap, `smolvla-base` public split export, synthetic
obs, `num_steps=10`.

> **Correction (2026-08-25, later):** these numbers were taken with clocks **NOT**
> pinned. `jetson-perf.service` was enabled but had been failing at every boot —
> its `ExecStart=/usr/bin/jetson_clocks` exits non-zero on a board whose GPU has not
> finished initialising, and nothing checked. `nvpmodel` still reported MAXN_SUPER, so
> the board looked configured while every clock floated. Re-measured on a genuinely
> pinned board (CPU 1728 MHz min=max, GPU 1020 MHz min=max, EMC 3199 MHz) the medians
> barely move but the spread collapses: the tuned config goes p50 134.8 -> 133.1 ms
> with p95 144.4 -> 133.8 ms, std 1.00 ms, and drift q4/q1 -2.0% -> -0.1%. The old
> "drift" was the DVFS governor ramping, exactly as `01-host-setup.md` warns, not
> thermals. Treat the per-step table below as sound in its *ratios* — every row shares
> the same unpinned condition — and see `RESULTS.md` for pinned absolutes.
Every step below was measured in isolation, and each one's output was checked against
the previous configuration before it was kept.

## The recipe

```bash
.venv-ort/bin/python -m bench ort-split --model smolvla-base \
    --bundle ~/bundles/smolvla-base-split \
    --tokenizer ~/bundles/smolvlm2-tokenizer \
    --precision fp16 --projectors gpu --iobinding --iters 100
```

| config | p50 ms | Hz | cpu cores busy | mJ/infer | parity vs previous |
|---|---:|---:|---:|---:|---|
| stock split (CPU projectors) | 229.8 | 4.35 | 2.86 | 3840 | — |
| `--projectors gpu` | 179.8 | 5.56 | 0.59 | 2962 | cosine 1.000000 |
| `+ --iobinding` | **134.8** | **7.35** | **0.53** | **2522** | **bit-identical** |

**1.70x faster than the stock split, 7.0x faster than the best PyTorch path
(torch-half16, 938 ms), and it gives back 2.33 of six CPU cores.** Chunk headroom at
30 fps goes from 3.1x to 12.0x. Cost: RSS 4162 -> 4632 MB.

## How many cameras? One. Two costs one more vision pass

Every number above is **one real camera** padded to the export's two slots
(`views=1`, `n_cam_slots=2`, prefix 177 = 2x64 img + 48 lang + 1 state), on the
**base** `lerobot/smolvla_base` weights with synthetic observations. A padded slot
costs zero vision passes — its embedding is computed once at load — but still occupies
its 64 prefix tokens, so a second *real* camera buys one extra vision pass and nothing
else:

| views | p50 ms | Hz | vision ms (calls) | prefill | decode |
|---|---:|---:|---:|---:|---:|
| 1 | 134.8 | 7.35 | 33.9 (1) | 12.2 | 62.1 |
| 2 | **174.0** | **5.68** | 68.2 (2) | 12.3 | 62.3 |

Confirmed empirically: prefill and decode do not move (12.2 -> 12.3, 62.1 -> 62.3);
the entire +39 ms is the second vision pass plus its numpy. **A two-camera rig like
`kaivuriprokkis` (D435i IR + colour) should budget ~174 ms, not 135.**

Do not read a parity verdict across different `--views`: two real cameras is a
different observation, not different arithmetic. Comparing `--views 2` against a
`--views 1` reference reports cosine 0.649, which means nothing. Compare like with
like — `--views 2` with and without `--iobinding` is bit-identical
(max abs diff 0.000e+00), which is the check that actually validates the code path.

Per-stage, 20 cycles (ms per inference):

| stage | stock | projectors gpu | + iobinding |
|---|---:|---:|---:|
| vision (TRT, x1) | 34.1 | 34.0 | 34.0 |
| prefill (TRT, x1) | 14.8 | 14.8 | 12.9 |
| **decode (TRT, x10)** | 109.0 | 102.1 | **65.6** |
| time_in (x10) | 36.4 | 4.0 | 3.8 |
| time_out (x10) | 20.8 | 3.4 | 3.2 |
| action_in + action_out (x10) | 4.4 | 7.9 | 5.9 |
| **wall** | **231.6** | **178.6** | **136.2** |

## What worked

**1. `--projectors gpu` — take it.** The four CPU projectors were 61.6 ms, 26.6% of
wall, and almost all of it was `time_in` (36.4) + `time_out` (20.8); `action_in`/
`action_out` were 4.4 ms combined. Moving them cuts wall 22.9% and `cpu_cores_busy`
from 2.86 to 0.59 — the number that matters when a 100 Hz control thread wants the
same six cores.

*Move all four or none.* Moving only `time_in`/`time_out` is **worse** than moving all
four (183.9 ms vs 178.6): `time_in`'s per-call cost goes 0.40 -> 1.08 ms when it has
CPU neighbours. Transfer/sync overhead dominates, which is what item 3 then exploits.

*TRT declines all four.* ORT logs `No graph will run on TensorRT execution provider`
and they run on the **CUDA EP** — while `get_providers()[0]` still cheerfully reports
`TensorrtExecutionProvider`. See the warning at the bottom of this file.

**3. IOBinding — the single biggest win, and free.** The stock loop re-feeds the KV
cache as numpy every step: 32 tensors, 7.2 MB, x10 = **72 MB of host->device copies per
inference** for data that is constant after prefill. Binding it to device once, and
having prefill write its KV straight to device rather than via the host, takes decode
from 101.2 -> 65.6 ms and wall from 181.6 -> 136.2. Output is **bit-identical**
(max abs diff 0.000e+00 over 8 chunks). Implemented in
`bench/backends/split_iobind.py`, applied to the live policy instance so the vendored
runtime stays byte-identical to what runs on the robot.

Two traps found while validating it, both now fixed and worth knowing if you port this:

- **The multi-camera path bypassed it entirely.** `_sample_actions_multiview` rebuilds
  the prefix itself and re-implements the denoise loop, so patching
  `policy.sample_actions` did nothing as soon as `--views > 1` — the first two-camera
  run was 215.7 ms with the flag on and no effect from it. The prefill+denoise core is
  now shared (`policy._iobind_prefill_denoise`), giving 212.1 -> 174.0 ms at two
  cameras, bit-identical.
- **`run_with_iobinding` was not timed.** It reaches the wrapped session through
  `_TimedSession.__getattr__`, so `--iobinding` runs reported `graph.decode` and
  `graph.prefill` as **0.00** and dumped ~76 ms into `python_numpy`. The wrapper now
  intercepts it; the breakdown reads decode 62.1 / prefill 12.2 / numpy 14.7.

## What did not work

**2. TRT build settings — the defaults are already right.** `TRT_OPT_LEVEL=5`
(workspace 1024 MB) **does not build on this board**. It failed the same way the
monolith does: a 318 MB constant-region allocation failure, then
`Error Code 10: Could not find any implementation for node`, with only 2 of 3 engines
built and the third heading for a silent CUDA-EP fallback. Killed after ~9 min rather
than waiting out the documented 85-minute failure. Level 5 explores more tactics per
node and each needs a working allocation — on 8 GB unified memory it runs out. The
vendored defaults (opt level unset = TRT 10's 3, workspace 1024 MB) stand.

**4. Precompute the time-embedding term — now pointless.** Worth doing only if
`time_in` was significant, and after item 1 it is 3.8 ms. The whole trick can save
~2 ms. Dropped.

## Gated on robot validation

**5. Fewer denoise steps.** Real latency, but it genuinely changes the actions — this
is the one item that alters policy behaviour rather than just cost:

| num_steps | p50 ms | Hz | cos_min vs 10-step | max abs diff |
|---|---:|---:|---:|---:|
| 10 | 179.8 | 5.56 | — | — |
| 5 | 116.7 | 8.52 | 0.981 | 1.95 (11.0% of range) |
| 4 | 103.9 | 9.60 | 0.960 | 15.0% of range |

Both fail the 0.999 gate, as a different ODE step count should. Base weights produce
meaningless actions, so nothing on this board can say whether the difference *matters* —
that needs the action<->motion correlation check against real episodes on the
fine-tuned checkpoint. Do not ship a reduced step count on a latency argument alone.

**6. The monolith A/B — not runnable here.** No monolithic SmolVLA ONNX exists locally
and it has to be exported on the Spark. The prior is strong that it will not TRT-build
(see `docs/03-backends.md`), and item 2 above is fresh evidence for the same mechanism
at much smaller scale.

## Smaller split — one fusion is worth it

Fusing the **four projectors + SiLU into the decode graph** turns 5 GPU calls per step
into 1, removing 40 host round trips per inference. Weight cost is trivial — projectors
are 6.4 MB against decode's 399 MB (+1.6%) — so by the build-peak formula in
`docs/03-backends.md` it still fits. Ceiling is modest now: ~13 ms of projector time
plus some in-loop numpy, so expect ~120 ms rather than a step change. The concat and
SiLU between `time_in` and `time_out` are numpy today and would have to become ONNX
ops, which is far cleaner at export time on the Spark than with `onnx.compose` here.

Do **not** fuse vision+prefill: 393 MB + 644 MB = 1.04 GB of weights puts the build
peak near 9 GB by that same formula, and it would save one round trip on a
once-per-inference path.

## Vision NaN-guard cleanup - measured, parity still gated

The Torch 2.12 two-slot export contains twelve
`Softmax -> IsNaN -> Where -> MatMul` attention paths. For finite image inputs the
`IsNaN` condition is always false, but the guards prevent TensorRT from selecting the
same fast vision graph as the older public export. Create a separate, validated copy
on the export workstation:

```bash
.venv-export/bin/python scripts/optimize_smolvla_vision.py \
    ~/bundles/smolvla-base-split-ours2 \
    ~/bundles/smolvla-base-split-ours2-no-nan-guard
```

The tool refuses to overwrite a destination, requires all twelve isolated patterns,
preserves the original external tensor layout, validates the graph, and requires
bit-identical CPU ORT output over five deterministic inputs. It also updates
`export_info.json` and `MANIFEST.sha256` before publishing the new directory.

On the pinned 8 GB Orin Nano, with two real cameras, FP16, GPU projectors and
IOBinding, this changed only the vision stage:

| graph | p50 ms | p95 ms | Hz | vision ms | VDD_IN W |
|---|---:|---:|---:|---:|---:|
| Torch 2.12 export | 189.89 | 190.86 | 5.248 | 91.234 | 18.13 |
| no-NaN-guard | **164.66** | **165.81** | **6.044** | **66.232** | 17.72 |
| no-NaN-guard repeat | **164.85** | **166.21** | **6.052** | **66.326** | 17.71 |

The repeat produced bit-identical action chunks. The optimized engine also passes the
strict comparison against the unmodified FP16 engine (0.534% of action range), but it
still **fails** the FP32 Torch certification threshold: 1.566% of range against the
1% gate. Keep it as a measured candidate, not a parity-certified default. Full signed
results are in `results/audit-patch-smolvla/`.

## Warning: `get_providers()[0]` is not what ran

ORT reports the session's provider *preference list*, not the provider that took the
graph. With `--projectors gpu`, all four projectors log `No graph will run on
TensorRT execution provider` and execute on the CUDA EP, yet the session still reports
`TensorrtExecutionProvider` first. **`configured_provider_priority_per_graph` in the result JSON inherits
this bug**, so `docs/05-runbook.md`'s advice to read that field to confirm a TRT engine
built is not sufficient. Confirm instead that an `.engine` file appeared in the cache
directory, or watch the ORT warnings during load.
