# Profile: where the time goes

Nsight Systems traces of every pure-TensorRT model on the bench board, 20 inferences
each after warmup, default denoising steps, the same synthetic observations as the
benchmark. Traced latency is within a few percent of an untraced run's. To repeat one:

```bash
nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop \
    --sample=none --cpuctxsw=none --cuda-graph-trace=node -o traces/xvla-base \
    .venv-ort/bin/python -m bench.tools.nsys_trace --model xvla-base \
    --bundle ~/bundles/xvla-base-split --chain graph --sidecar traces/xvla-base.trace.json
nsys export --type sqlite -o traces/xvla-base.sqlite traces/xvla-base.nsys-rep
python -m bench.tools.nsys_trace --summarize traces/xvla-base.sqlite
```

## The default chain

`--chain graph`: the whole inference on the GPU, replayed as a CUDA graph, one
synchronize per call.

| model | p50 under nsys | GPU kernels | not kernels | GPU busy |
|---|---:|---:|---:|---:|
| smolvla-base | 150.96 ms | 145.6 ms | 5.4 ms | 96 % |
| evo1-libero | 361.41 ms | 338.8 ms | 22.7 ms | 94 % |
| xvla-base | 384.64 ms | 362.0 ms | 22.7 ms | 94 % |
| groot-n16-base | 288.02 ms | 272.8 ms | 15.2 ms | 95 % |
| groot-n17-base | 288.1 ms | 277.3 ms | 11.0 ms | 96 % |
| pi05-libero | 475.76 ms | 441.7 ms | 34.3 ms | 93 % |

What is not kernel time is mostly host image preprocessing before the upload (SmolVLA's
512×512 pad is cheapest; π0.5's two-camera resize is the most).

## Per stage

`--chain host` runs the same engines with a synchronize and a copy back after each one,
which is what lets the trace attribute time to stages. `launch+sync+py` is the part of
an engine call the GPU spends idle; the graph chain removes most of it and all of the
copies. `weights GB/s` is the engine's size over its kernel time; the board's DRAM peaks
at ~102 GB/s, so a stage near it is bandwidth-bound and one far below it compute-bound.
π0.5 has no host chain.

### smolvla-base: 20 traced inferences, p50 193.52 ms under nsys

| stage | calls/infer | wall ms | GPU kernels ms | copies ms | launch+sync+py ms | engine MB read/infer | weights GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision | 2 | 90.6 | 86.4 | 0.6 | 3.6 | 395 | 5 |
| state_proj | 1 | 0.4 | 0.0 | 0.0 | 0.4 | 0 | 14 |
| prefill | 1 | 15.3 | 10.7 | 0.4 | 4.2 | 298 | 28 |
| action_in | 10 | 3.2 | 0.1 | 0.1 | 3.0 | 1 | 10 |
| time_in | 10 | 3.6 | 0.6 | 0.2 | 2.7 | 42 | 69 |
| time_out | 10 | 3.2 | 0.3 | 0.2 | 2.7 | 21 | 61 |
| decode | 10 | 63.0 | 52.4 | 0.6 | 10.0 | 1981 | 38 |
| action_out | 10 | 3.0 | 0.1 | 0.1 | 2.8 | 1 | 7 |
| host glue between engines | | 11.6 | | | | | |
| **total** | | **193.8** | **150.7** | 2.1 | | | |

GPU kernels busy 78 % of the wall time. Per inference: 54 stream synchronizations, 21.5 MB host->device, 13.6 MB device->host.

### evo1-libero: 20 traced inferences, p50 427.14 ms under nsys

| stage | calls/infer | wall ms | GPU kernels ms | copies ms | launch+sync+py ms | engine MB read/infer | weights GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision | 4 | 187.1 | 175.5 | 6.7 | 4.9 | 623 | 4 |
| language | 3 | 39.2 | 34.7 | 0.8 | 3.8 | 419 | 12 |
| action_context | 1 | 9.9 | 2.3 | 1.4 | 6.2 | 28 | 12 |
| action_step | 32 | 129.6 | 105.7 | 3.0 | 20.8 | 4381 | 41 |
| action_output | 32 | 37.0 | 28.2 | 0.4 | 8.4 | 2707 | 96 |
| host glue between engines | | 24.7 | | | | | |
| **total** | | **427.5** | **346.4** | 12.3 | | | |

GPU kernels busy 81 % of the wall time. Per inference: 72 stream synchronizations, 78.6 MB host->device, 72.2 MB device->host.

### xvla-base: 20 traced inferences, p50 416.14 ms under nsys

| stage | calls/infer | wall ms | GPU kernels ms | copies ms | launch+sync+py ms | engine MB read/infer | weights GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision | 4 | 95.1 | 89.9 | 1.2 | 4.1 | 735 | 8 |
| text_encoder | 3 | 6.8 | 5.4 | 0.1 | 1.3 | 418 | 78 |
| cond | 1 | 0.8 | 0.1 | 0.1 | 0.5 | 5 | 35 |
| denoise | 40 | 291.4 | 265.9 | 2.9 | 22.6 | 6072 | 23 |
| host glue between engines | | 21.8 | | | | | |
| **total** | | **415.8** | **361.3** | 4.2 | | | |

GPU kernels busy 87 % of the wall time. Per inference: 48 stream synchronizations, 42.7 MB host->device, 40.9 MB device->host.

### groot-n16-base: 20 traced inferences, p50 348.26 ms under nsys

| stage | calls/infer | wall ms | GPU kernels ms | copies ms | launch+sync+py ms | engine MB read/infer | weights GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision | 5 | 119.6 | 111.2 | 4.1 | 4.3 | 859 | 8 |
| llm | 8 | 69.5 | 61.2 | 3.8 | 4.5 | 1616 | 26 |
| cond | 1 | 1.3 | 0.1 | 0.4 | 0.8 | 4 | 26 |
| kv | 1 | 21.5 | 7.6 | 8.2 | 5.7 | 202 | 26 |
| dit | 32 | 122.2 | 94.2 | 7.6 | 20.4 | 6758 | 72 |
| host glue between engines | | 16.7 | | | | | |
| **total** | | **350.8** | **274.4** | 24.1 | | | |

GPU kernels busy 78 % of the wall time. Per inference: 47 stream synchronizations, 119.9 MB host->device, 115.3 MB device->host.

### groot-n17-base: 20 traced inferences, p50 368.54 ms under nsys

| stage | calls/infer | wall ms | GPU kernels ms | copies ms | launch+sync+py ms | engine MB read/infer | weights GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision | 4 | 72.2 | 65.7 | 2.2 | 4.4 | 814 | 12 |
| llm | 8 | 101.1 | 88.3 | 6.8 | 6.0 | 1618 | 18 |
| cond | 2 | 25.0 | 21.6 | 1.6 | 1.8 | 408 | 19 |
| kv | 1 | 36.7 | 11.0 | 19.8 | 5.9 | 202 | 18 |
| dit | 32 | 125.4 | 94.8 | 10.4 | 20.2 | 6757 | 71 |
| host glue between engines | | 15.6 | | | | | |
| **total** | | **375.9** | **281.4** | 40.8 | | | |

GPU kernels busy 75 % of the wall time. Per inference: 47 stream synchronizations, 160.0 MB host->device, 147.7 MB device->host.

## Reading it

- With the graph chain the GPU is busy 93–96 % of each call. What remains is kernel
  time: faster now means fewer bytes or less arithmetic, not less overhead.
- Vision is compute-bound in every model (4–12 GB/s of weights). In SmolVLA, the FP32
  softmax the mixed-FP16 pass keeps takes about 19 ms of the 86 ms vision time; letting
  it run in FP16 did not help (151 ms against 149 ms end to end, with a slightly larger
  action error), so the export keeps it FP32.
- GR00T's DiT reads its weights at ~72 GB/s, near the bandwidth limit: its step graphs
  no longer carry the AdaLN modulation (computed at load) or the cross-attention K/V
  (once per observation), which took it from 128 to 94 ms of kernels.
- EVO1's action output (96 GB/s) is bandwidth-bound; X-VLA's denoiser (23 GB/s) runs all
  262 tokens every step and is compute-bound.
- π0.5's time is GEMMs, most of it in the 18 language layers.

## Smaller-model follow-ups (2026-10-10)

A local runtime review found four small fixes in `bench/vendor/trt_device.py`:

| area | change | expected benefit |
|---|---|---|
| CUDA graphs | Destroy the source graph after instantiation, including error paths; retire the old executable before EVO1 image-layout or SmolVLA camera-count recapture. | Avoid accumulating unused graph resources during layout changes. |
| EVO1 prompt cache | Compare the validity mask as well as token IDs; keep copies of both. | Correct masks when IDs are unchanged but valid positions change, including caller arrays mutated in place. |
| SmolVLA prompt cache | Refresh prompt rows and masks if no cache key is supplied. Explicit task keys retain caching. | Avoid stale language data for callers using the optional-key interface. The benchmark already supplies a task key. |
| X-VLA prompt upload | Upload IDs only when their contents change; retain a snapshot to detect mutation. | Remove one small H2D operation for repeated tasks. This is unlikely to materially change end-to-end latency. |

CUDA separates the source graph from its executable, so releasing the source after
instantiation preserves replay; see the [NVIDIA CUDA graph lifecycle examples](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).
All 30 unit tests passed locally and on the Nano. On-board checks also passed the
full-chunk fixture gates for SmolVLA, X-VLA, EVO1, GR00T N1.6, GR00T N1.7 and compact
π0.5. The three smaller models ran paired 60-second timing windows against commit
`6795648`, using identical cached engines, observations, noise, cameras and steps:

| model | baseline p50 ms | updated p50 ms | updated p95 ms | achieved Hz | updated calls |
|---|---:|---:|---:|---:|---:|
| smolvla-base | 148.83 | 148.80 | 149.89 | 6.7136 | 403 |
| xvla-base | 382.81 | 383.29 | 384.93 | 2.6079 | 157 |
| evo1-libero | 358.81 | 359.65 | 360.52 | 2.7790 | 167 |

All eight saved action chunks per model were **bit-for-bit identical** to baseline.
The six paired windows completed 1,455 calls. The timings support correctness and
no material runtime regression; they do not establish a speedup. GR00T and π0.5 ran
five measured calls each as shared-helper smoke tests, rather than paired timing runs.

Each smaller model also passed 40 stress calls against its TRT host reference:

- SmolVLA alternated one/two cameras and changed language without a cache key. Across
  41 recaptures, 41 old executables were retired; peak live executables stayed at one
  and live source graphs returned to zero. Process RSS grew 4 KiB during stress.
- EVO1 changed only the validity mask, verified both masks downloaded from the GPU,
  and alternated image-token copy layouts. It also retired all 41 replaced executables,
  retained no source graphs, and had unchanged RSS. All checked actions were identical
  to the host chain.
- X-VLA reused unchanged prompt IDs, detected an in-place change, and alternated prompt
  switches with repeated prompts. Its RSS grew about 2.45 MiB over 40 calls; this short
  check does not establish long-session cache behavior.

All three also passed an intentional Python enqueue exception inside a real CUDA
capture: capture ended, the source graph was released, and the original exception was
preserved. Worst host-reference action error was 0.0864 % of range (SmolVLA), below the
existing 1 % gate.

These runs used the board's existing MAXN_SUPER setting and locked clocks. OC3 event
counters increased in both baseline and updated timing windows; retain that power
condition when interpreting the numbers. The monitor reported no active thermal
cooling clamps. Repeat the standard 300-second runs before drawing endurance or
thermal conclusions. [Raw runs, probes and reproduction scripts](../results/runtime-updates-20261010/)
and the [machine-readable summary](../results/runtime-updates-20261010/summary.json)
are saved beside the existing results.

The next performance experiments should follow that baseline:

1. **SmolVLA vision:** inspect the dominant kernels before writing Triton replacements.
   A two-camera batch export could be worth testing against the current two serial
   calls, provided its build and scratch fit. The earlier FP16 softmax experiment
   already failed to improve end-to-end time, so retain the current precision choice.
2. **X-VLA denoising:** inspect attention, normalization and GEMM tactics across the
   262-token sequence. The exporter already offers `--fuse-denoise-interpolation`,
   but the pure-TRT runtime currently rejects that contract. Supporting it would
   remove the separate interpolation engine and intermediate `x_t` buffer; measure
   the gain rather than assuming ten tiny launches dominate a captured graph.
   Conditioning and soft-prompt rows pass through self-attention each step, so their
   deeper hidden states cannot simply be cached like cross-attention K/V.
3. **EVO1:** focus kernel work on vision and `action_step`. The output head's measured
   96 GB/s weight rate leaves little room for a generic faster-GEMM replacement.
   Cross-attention K/V is already computed once per observation. Keep 32 steps for
   equivalent-work comparisons; reducing steps is a separate policy-quality test.

For the first board check, use each backend's existing full-chunk fixture gate and
saved-observation comparisons against the current TRT baseline, then repeat sustained
runs with identical cameras, steps and power settings. Include prompt switches and
supported camera/layout changes when checking resource growth. No new PyTorch timing
comparison is needed. Broader buffer/event teardown and bounded task caches remain
separate lifecycle work; the graph fixes do not provide complete backend cleanup.

## Initial SmolVLA Triton results (2026-10-10)

Desktop triton-kernel-lab softmax examples passed real SM87 compilation and
operator checks. Two TensorRT AOT plugin candidates also passed complete-action
fixtures and eight saved-chunk comparisons against a freshly rebuilt TRT control:

| vision implementation | p50 ms | p95 ms | achieved Hz | saved-chunk max error, % of control range |
|---|---:|---:|---:|---:|
| original TRT graph, rebuilt control | 148.65 | 149.79 | 6.7189 | — |
| Triton softmax + surrounding casts | 170.95 | 172.13 | 5.8424 | 0.108 |
| Triton mask + casts + softmax + NaN guard | 151.56 | 152.34 | 6.5916 | 0.147 |

These are separate 60-second, two-view, ten-step graph runs on the Nano. All
other engines are byte-identical to the baseline. The compiled engines run in
the existing TRT environment with neither Torch nor Triton installed.

The standalone FP16-I/O/FP32-math row softmax is 5.52x faster than a standalone
TRT Cast/Softmax/Cast graph, but the production engine already fuses a wider
sequence. Inspection shows 116 vision layers in control, 212 in the narrow
replacement, and 145 in the broader replacement. Neither complete policy is
faster, so both remain opt-in experiments. The next candidate should fuse QK,
mask/softmax and PV, avoiding the attention matrix rather than replacing its
softmax alone.

FP32 accumulation, exported FP16 Add rounding and NaN-to-zero behavior are
preserved. Random, masked, large-rounding and infinity probes pass. Existing
MAXN_SUPER/clocks were retained; OC3 counters rose in all windows. The tests
establish initial integration and numerical feasibility, not endurance or task
quality. [Implementation and reproduction](../experiments/smolvla_triton/) and
[raw results, source snapshots and summary](../results/smolvla-triton-20261010/)
are retained locally. The [archive command](05-runbook.md#8-preserve-remote-measurements-locally)
keeps timestamped copies of remote measurement files before later overwrites.

## SmolVLA expert follow-up (2026-10-10)

The subsequent aligned Triton vision-attention candidate passed a paired
five-minute comparison: 128.27 ms p50 against 148.88 ms for the original-vision
control. A fresh 20-inference graph trace of that improved policy measures
130.5 ms under Nsight, including 125.1 ms of kernels. The remaining expert
attention already uses TensorRT's fused MHA kernels, about 5.6 ms across the
ten denoising steps. Most residual/RMS normalization chains are also fused.

The larger remaining expert target is the gated feed-forward projection. It
runs sixteen times per denoising step: a 50×720 input is projected to gate/up
branches of width 2048 before SiLU and multiplication. The graph trace charges
about 11.9 ms to these projections per inference. A separate TensorRT layer
profiler confirms their importance; its synchronization overhead makes its
totals unsuitable for a throughput comparison.

The first direct dual-GEMM Triton sweep mostly loses to TensorRT. Interleaving
the two weight matrices permits a single larger GEMM with the gate applied
before the output store. At the actual expert shapes with synthetic weights,
the selected kernel measures 0.0746 ms against 0.0829 ms for an isolated TRT
graph. With the real weights, all twelve full padded-action stress cases pass,
with worst error 0.0503% of the aligned control's action range.

The initial identical-work 60-second windows measure 127.88 ms for the previous
aligned policy, 127.91 ms for a rebuilt expert control and 126.82 ms for the
packed expert candidate. Mean denoising time falls from 46.794 to 45.722 ms;
GPU clocks remain comparable. Every non-decoder engine has identical bytes,
and runtime execution has neither Torch nor Triton installed.

Paired 300-second windows confirm 126.67 ms p50 / 127.63 ms p95 for the packed
candidate against 128.10 / 129.10 ms for control (1.12% lower median latency).
Mean denoising time is 45.716 versus 46.852 ms. The sustained saved-action
comparison passes across eight observations, with maximum error 0.024% of
reference range. The candidate shows essentially zero quartile drift. GPU
clocks are comparable and neither run has active thermal cooling clamps;
both retain the existing OC3 over-current events. Candidate mean whole-board
RAM use is approximately 1.80 GB, about 66 MB above control.

[Implementation, reproduction and sustained results](../experiments/smolvla_triton/#expert-gated-projections)
and [raw trace, operator sweeps and parity evidence](../results/smolvla-expert-20261010T1041Z/)
are retained separately from the standard backend measurements.

## SmolVLA runtime memory follow-up (2026-10-10)

Fresh-process probes found 18–21 MiB of freed glibc heap after initialization.
Returning those pages once preserves exact padded actions. Loading the saved
Rust tokenizer directly avoids Transformers at runtime; exact comparisons pass
for 215 prompts, including left truncation, Unicode and special tokens. An
eight-entry prompt cache prevents memory growth across unbounded distinct tasks.

The benchmark's environment recorder itself imported unused frameworks just for
version strings. It now reads installed package metadata without importing them,
while preserving versions from modules already loaded. Twenty relevant tests
pass on the host and Nano. Corrected 60-second runs measure 127.97 ms / 1803 MB
board RAM for aligned vision with the original expert and standard tokenizer,
127.86 ms / 1759 MB for its lean runtime, and 126.86 ms / 1791 MB for the lean
packed expert. The latter is the balanced experimental reference: approximately
1 ms faster than the lower-RAM option for 32 MB more sampled board RAM.

[Runtime flags and reproduction](../experiments/smolvla_triton/#ram-and-speed-on-the-8-gb-nano)
and [memory audit evidence](../results/smolvla-memory-20261010/) retain the
alternatives. CPU RSS and CUDA allocations overlap on the Nano; do not sum them
or treat disk-cache size as resident RAM. Numerical checks remain separate from
robot task success.

The selected packed-expert/lean-runtime five-minute check passes at 126.78 ms
p50, 127.71 ms p95 and 7.8768 achieved Hz, with -0.2% quartile latency drift.
Whole-board mean RAM is 1815 MB and process mean RSS 1458.6 MB. Saved actions
are bit-identical to the same expert cache before memory changes. The runtime
does not import Torch, Triton or Transformers. All alternatives and raw evidence
remain archived; the main README shows only the selected SmolVLA configuration.
