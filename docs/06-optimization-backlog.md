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
`providers_per_graph` in the result records where each one actually landed, and
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

Run it before the Tether backend: Tether's fast path is also a monolithic ONNX, so
knowing what the monolith does on this board *independently of Tether's runtime*
separates "Tether is slow" from "the monolith cannot build here".

---

## Why this is worth writing down

A per-graph breakdown of a split VLA on an 8 GB Orin, plus a monolithic build result on
the same board, is not published anywhere. Keep the numbers per change, keep the
failures, and the backlog turns into a result rather than a to-do list.
