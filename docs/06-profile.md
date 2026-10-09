# Profile: where the time goes

Nsight Systems traces of every pure-TensorRT model on the bench board, 20 inferences
each after warmup, default denoising steps, the same synthetic observations as the
benchmark. Traced latency is 3–5 % above an untraced run's. To repeat one:

```bash
nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop \
    --sample=none --cpuctxsw=none -o traces/xvla-base \
    .venv-ort/bin/python -m bench.tools.nsys_trace --model xvla-base \
    --bundle ~/bundles/xvla-base-split --sidecar traces/xvla-base.trace.json
nsys export --type sqlite -o traces/xvla-base.sqlite traces/xvla-base.nsys-rep
python -m bench.tools.nsys_trace --summarize traces/xvla-base.sqlite
```

`launch+sync+py` is the part of an engine call the GPU spends idle: enqueue, the
stream synchronization after every engine, and Python. `weights GB/s` is the engine's
size over its kernel time; the board's DRAM peaks at ~102 GB/s, so a stage near it is
bandwidth-bound and one far below it is compute-bound.

## smolvla-base: 20 traced inferences, p50 193.77 ms under nsys

| stage | calls/infer | wall ms | GPU kernels ms | copies ms | launch+sync+py ms | engine MB read/infer | weights GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision | 2 | 90.1 | 86.0 | 0.6 | 3.5 | 395 | 5 |
| state_proj | 1 | 0.3 | 0.0 | 0.0 | 0.3 | 0 | 14 |
| prefill | 1 | 15.4 | 10.7 | 0.4 | 4.4 | 298 | 28 |
| action_in | 10 | 3.2 | 0.1 | 0.1 | 3.0 | 1 | 10 |
| time_in | 10 | 3.6 | 0.6 | 0.2 | 2.7 | 42 | 69 |
| time_out | 10 | 3.3 | 0.3 | 0.2 | 2.8 | 21 | 61 |
| decode | 10 | 63.3 | 52.2 | 0.6 | 10.5 | 1981 | 38 |
| action_out | 10 | 3.1 | 0.1 | 0.1 | 2.9 | 1 | 7 |
| host glue between engines | | 11.6 | | | | | |
| **total** | | **193.9** | **150.1** | 2.1 | | | |

GPU kernels busy 77 % of the wall time. Per inference: 54 stream synchronizations, 21.5 MB host->device, 13.6 MB device->host.

## evo1-libero: 20 traced inferences, p50 426.52 ms under nsys

| stage | calls/infer | wall ms | GPU kernels ms | copies ms | launch+sync+py ms | engine MB read/infer | weights GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision | 4 | 187.1 | 175.5 | 6.7 | 5.0 | 623 | 4 |
| language | 3 | 39.1 | 34.6 | 0.8 | 3.8 | 419 | 12 |
| action_context | 1 | 9.9 | 2.3 | 1.4 | 6.2 | 28 | 12 |
| action_step | 32 | 129.0 | 105.6 | 3.1 | 20.4 | 4381 | 42 |
| action_output | 32 | 36.8 | 28.2 | 0.4 | 8.2 | 2707 | 96 |
| host glue between engines | | 24.7 | | | | | |
| **total** | | **426.6** | **346.2** | 12.2 | | | |

GPU kernels busy 81 % of the wall time. Per inference: 72 stream synchronizations, 78.6 MB host->device, 72.2 MB device->host.

## xvla-base: 20 traced inferences, p50 414.02 ms under nsys

| stage | calls/infer | wall ms | GPU kernels ms | copies ms | launch+sync+py ms | engine MB read/infer | weights GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision | 4 | 95.3 | 90.1 | 1.1 | 4.0 | 735 | 8 |
| text_encoder | 3 | 6.8 | 5.4 | 0.1 | 1.2 | 418 | 77 |
| cond | 1 | 0.7 | 0.1 | 0.1 | 0.5 | 5 | 35 |
| denoise | 40 | 289.4 | 264.5 | 2.9 | 22.0 | 6072 | 23 |
| host glue between engines | | 22.0 | | | | | |
| **total** | | **414.2** | **360.1** | 4.2 | | | |

GPU kernels busy 87 % of the wall time. Per inference: 48 stream synchronizations, 42.7 MB host->device, 40.9 MB device->host.

## groot-n16-base: 20 traced inferences, p50 357.21 ms under nsys

| stage | calls/infer | wall ms | GPU kernels ms | copies ms | launch+sync+py ms | engine MB read/infer | weights GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision | 5 | 120.2 | 112.0 | 4.0 | 4.2 | 859 | 8 |
| llm | 8 | 69.5 | 61.3 | 3.7 | 4.4 | 1616 | 26 |
| cond | 1 | 1.4 | 0.1 | 0.4 | 0.8 | 4 | 26 |
| time | 4 | 2.9 | 0.1 | 0.0 | 2.8 | 0 | 5 |
| dit | 44 | 147.0 | 128.2 | 1.5 | 17.4 | 8839 | 69 |
| host glue between engines | | 16.5 | | | | | |
| **total** | | **357.4** | **301.7** | 9.6 | | | |

GPU kernels busy 84 % of the wall time. Per inference: 62 stream synchronizations, 59.1 MB host->device, 56.2 MB device->host.

## groot-n17-base: 20 traced inferences, p50 374.46 ms under nsys

| stage | calls/infer | wall ms | GPU kernels ms | copies ms | launch+sync+py ms | engine MB read/infer | weights GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision | 4 | 72.5 | 65.9 | 2.2 | 4.4 | 814 | 12 |
| llm | 8 | 101.3 | 88.4 | 6.8 | 6.1 | 1618 | 18 |
| cond | 2 | 24.9 | 21.4 | 1.6 | 1.9 | 408 | 19 |
| time | 4 | 2.7 | 0.1 | 0.0 | 2.6 | 0 | 4 |
| dit | 44 | 158.2 | 138.7 | 1.4 | 18.1 | 8839 | 64 |
| host glue between engines | | 15.9 | | | | | |
| **total** | | **375.6** | **314.6** | 12.0 | | | |

GPU kernels busy 84 % of the wall time. Per inference: 62 stream synchronizations, 73.4 MB host->device, 62.6 MB device->host.

## Reading it

- GPU kernels are 77–87 % of the wall time. The rest — per-engine synchronization,
  the copies between engines and the host glue (preprocessing, numpy between calls) —
  is what a device-resident chain with CUDA Graph replay can remove: roughly 11–19 % of
  each model's latency, more for the ones with many small calls (SmolVLA, EVO1).
- Vision is compute-bound in every model (4–12 GB/s of weights). In SmolVLA, the FP32
  softmax that the mixed-FP16 pass keeps takes 19 ms of the 86 ms vision time; the
  vision tower does not get TensorRT's fused attention kernel, the decoder does.
- GR00T's DiT (64–69 GB/s) and EVO1's action output (96 GB/s) read their weights near
  the bandwidth limit: those stages only get faster by reading fewer bytes. X-VLA's
  denoiser (23 GB/s) runs all 262 tokens every step and is compute-bound.
