# 4. What each number means, and what it does not

Every run writes one JSON to `results/`. `python -m bench report results` turns a
directory of them into the tables in `RESULTS.md`. The JSON always holds more than the
table shows.

## The shape of a run

```
load          construct the backend, load weights/engines      -> load_s
first infer   the very first call, kept separate               -> first_infer_ms
warmup        N calls, discarded
idle window   model resident, nothing running                  -> the baseline
measure       M calls back to back, monitored                  -> the numbers
```

The **idle window is what makes the footprint numbers mean anything**. A backend's
resident cost (weights, engines, CUDA arenas) and its running cost are different
questions, and on an 8 GB unified-memory board that also has to host a robot control
stack, both matter. A single "RAM" figure answers neither.

## Latency

`p50` / `p95` / `max`, plus `quartile_means_in_order` and `drift_q4_vs_q1_pct`.

Use p95, not the mean. And read the drift, which exists because a short run cannot tell
you whether latency holds. Whether this board's clocks hold under a sustained VLA load
at MAXN is **not measured yet** — `--duration-s 300` is how it gets answered, and
`tj max °C` next to the drift column is what makes the answer readable. Until that run
exists, treat every latency figure here as a short-run figure.

`first_infer_ms` is separated out but never hidden. A lazy TensorRT build, a cuDNN
autotune or CUDA context creation lands there — and on a board where `/tmp` clears at
boot, it is the difference between a five-second start and a five-minute one.

### What latency is *not*

It is not the control rate. SmolVLA emits a **chunk** of actions authored at the
dataset fps; the controller plays them open-loop while the next inference runs. The
reference PyTorch demo everyone quotes as "10 Hz on an Orin Nano" is running its
*observation* loop at 10 Hz, not its inference. A 400 ms inference is entirely
compatible with a 100 Hz valve loop.

So the report also carries the figure that does bite:

```
steps_consumed_per_inference = mean_latency_s * fps
chunk_headroom_x             = chunk_size / steps_consumed
```

At 30 fps and a chunk of 12, a 400 ms inference consumes 12 steps — headroom 1.0, the
plan runs dry exactly as the next one lands. Below 1× the controller falls through to
hold-and-decay between plans. **That** is the threshold worth optimising against, not
a Hz number.

## CPU — the metric this comparison exists for

`cores_busy` is per-process CPU over the measurement window: **1.0 means one of the
six Orin Nano cores is gone**, and the robot control stack cannot have it. It is
reported for both the idle and the load window, so the difference is what inference
itself takes rather than what merely sitting there costs.

This is where the backends genuinely differ. The split-ONNX path runs six of its nine
graphs on the CPU execution provider and does the entire flow-matching loop in numpy —
four CPU graphs × 10 steps plus the Euler update. "It runs on the GPU" is a claim
about three graphs, not about the pipeline. The `latency_breakdown_ms` fields
`graphs_gpu`, `graphs_cpu` and `python_numpy` split it explicitly.

Per-process CPU is taken from `/proc/<pid>/stat` (utime+stime deltas over wall time)
and follows child processes, because `tether serve` puts the model in another process
entirely.

## RAM

Three different numbers, deliberately:

| field | means |
|---|---|
| `rss_after_load_mb` | resident set of the backend process(es) once loaded |
| `process.windows.load.rss_mb.max` | peak while inferring |
| `system.windows.load.ram_used_mb` | whole-board usage from tegrastats |

On a unified-memory Jetson the system figure is the one that decides whether
something gets OOM-killed, because the GPU allocations come out of the same pool.
`delta_load_minus_idle.ram_used_mb` is the running cost on top of resident.

## GPU

`GR3D_FREQ` from tegrastats, as a percentage. Treat it as an occupancy hint, not a
throughput measure — it says a kernel was resident, not that it was efficient. A
backend at 95% GPU and 0.2 cores of CPU is the shape this project wants; 60% GPU and
1.5 cores is not, whatever the latency says.

## Power and energy

Orin reports rails as `VDD_IN 4152mW/4152mW` — instantaneous / tegrastats' own running
average. The harness uses the instantaneous value. `VDD_IN` is the whole board at the
barrel jack; `VDD_CPU_GPU_CV` and `VDD_SOC` split it.

`mJ/infer` integrates VDD_IN over the measurement window and divides by the inference
count. **This is the honest cross-backend power comparison**: 25 W for 100 ms beats
15 W for 500 ms, and a plain watt figure inverts that ranking. All runs are at
MAXN_SUPER, so a lower watt figure means the board found nothing to do, not that the
backend is frugal.

`tj max °C` comes along for the ride, and should be read next to the drift column.

## Parity

Speed is only interesting if the actions survive. Because every backend is fed the
same seeded observation **and the same flow-matching noise draw**, action chunks line
up element by element and cosine means what it looks like it means.

| verdict | when |
|---|---|
| `PASS` | cosine ≥ 0.999, max abs diff ≤ 1% of the reference's action range, all finite |
| `FAIL` | anything else, with identical noise |
| `PLAUSIBLE` / `SUSPECT` | distribution-only comparison — see below |

Cosine hides a scale error, so an absolute difference is reported too — and an absolute
difference means nothing without knowing the action range. Different policies use
different action spaces (joystick rates in [-1, 1], normalized joint targets, a 20-dim
ee6d pose), so the difference is normalized against the **reference run's own observed
action range** rather than an assumed [-1, 1]. `max_abs_diff_pct_of_range` therefore
means the same thing across models: 1% is one percent of the span the reference policy
actually commands.

**The tether backend cannot be certified this way.** An HTTP server draws its own
noise, so its chunks integrate a different ODE and only a distribution comparison is
honest: per-dimension mean and std shifts, which catch a gross failure (wrong scale,
saturated axis, dead dimension) but cannot prove numerical agreement. The output says
so rather than papering over it.

Why parity is a first-class metric here and not a footnote: the Orin is compute 8.7,
FP16 is the only fast reduced precision it has, and blanket FP16 on SmolVLA is exactly
what collapsed the SigLIP vision tower to cosine 0.805 on Blackwell — 730 constants in
the vision attention exceed FP16's exponent range, including literal `inf` mask values.
The TensorRT path avoids that by keeping layer norms in FP32 and letting rejected ops
fall back to FP32 on the CUDA EP. A naive `.half()` does not. If a backend is fast and
wrong, this is what catches it.

## Fairness controls

- **Same observations.** Seeded per index, so backends can be run in any order.
- **Same preprocessing.** Resize, tokenize and MEAN_STD normalization are done by the
  harness, identically, using the bundle's own `stats.json` and `tokenizer/` — so a
  parity difference can only come from the runtime. Timed separately as
  `preprocess`. The exception is `tether`, which takes a raw frame and does its own;
  it is flagged `preprocess_owned: false`.
- **Same noise**, where injectable.
- **Same denoise steps and chunk size**, read from the export bundle rather than
  assumed.
- **Board state recorded** — `nvpmodel`, clocks, L4T, package versions, repo sha — in
  every JSON.

## The transport caveat for tether

`total_ms` for the tether backend is the **client roundtrip**: PNG encode, JSON, HTTP
over loopback, decode. That is a real cost if you deploy it that way, but it is not
the same quantity as the other backends' in-process `total`. The breakdown separates
`encode_request`, `roundtrip` and — when the server reports one — `server`. Compare
`server` against the others' `total` for a model-to-model figure, and use the
roundtrip when asking what a control loop would actually see. Tether also offers a
ZMQ transport and a ROS2 bridge; if the HTTP overhead turns out to dominate, that is
the next thing to measure, and it is worth saying so in the writeup rather than
quoting the roundtrip as if it were inference.
