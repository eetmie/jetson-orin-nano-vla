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
