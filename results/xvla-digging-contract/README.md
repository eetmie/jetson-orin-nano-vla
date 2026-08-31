# Fine-tuned X-VLA contract smoke deployment

Date: 2026-08-28

Checkpoint: `outputs/digging/ir/checkpoints/000250/pretrained_model` from
`xvla-spark-finetune`. This is the 250-step throughput/configuration smoke checkpoint,
not a robot-quality candidate. Task conditioning was `move the sand to the container`.

The schema-v2 split bundle fixes the deployment boundary to:

- one real camera in a three-slot model;
- 3-D raw state, normalized with the saved LeRobot processor, then padded to 20;
- 20-D normalized model action, trimmed to four real axes, then unnormalized;
- the exact local BART tokenizer and full checkpoint tree hash.

| run | p50 ms | p95 ms | Hz | load s | output |
|---|---:|---:|---:|---:|---|
| PyTorch FP32 / LeRobot 0.6.1 | 2090.15 | 2106.30 | 0.48 | 39.62 | 50 × 4 |
| ORT + TensorRT FP16 | 348.81 | 353.74 | 2.87 | 364.30 cold build/load | 50 × 4 |
| ORT + TensorRT FP16 repeat | 355.87 | 362.90 | 2.80 | 62.53 warm-cache load | 50 × 4 |

All 12 split graphs reported `TensorrtExecutionProvider`. TensorRT is approximately
6.0× faster than the stock FP32 reference by p50 latency.

Strict parity uses eight synthetic observations (indices 3–10), exact signed raw inputs,
and identical seeded 20-D noise. Against PyTorch, both TensorRT runs pass with minimum
cosine 0.9999994, maximum absolute physical-action error 0.001221, and maximum error
0.083% of the reference range. The two fresh TensorRT processes are bit-identical.

Local CPU FP32 parity additionally passes at cosine 1.000000 for conditioning tokens,
the padded 20-D model action, the normalized four-axis action, and the physical
four-axis action.

The canonical real-IR fixture freezes the middle frame, aligned raw 3-D state, and
recorded four-axis target from held-out episodes 5/15/25/35/45/55/65/75. Its tree ID is
`ad21c530d3b2dd638f776ba5dc3960f1d35cb6e2550e63eede03245587fbcb06`.

| real-IR run | p50 ms | p95 ms | achieved Hz | load s | provider |
|---|---:|---:|---:|---:|---|
| Jetson PyTorch FP32 | 2079.09 | 2084.24 | 0.481 | 42.33 | CUDA |
| Spark ORT FP32 | 8668.85 | 9041.67 | 0.115 | 8.16 | CPU |
| Jetson ORT + TensorRT FP16 | 359.92 | 365.97 | 2.792 | 25.11 | TRT |
| Jetson ORT + TensorRT FP16 repeat | 348.42 | 362.40 | 2.848 | 26.33 | TRT |

All four runs share the exact comparison signature, eight signed input hashes, and
eight seeded-noise hashes. CPU ORT versus Torch has cosine minimum 1.000000 and maximum
physical-action error 0.0003103 (0.021% of range). TensorRT versus Torch has cosine
minimum 0.9999763 and maximum error 0.005809 (0.394% of range). Both fresh TensorRT
processes produce bit-identical saved 50 x 4 chunks. The complete machine-readable
verdict is [`real-ir-parity.json`](real-ir-parity.json).

Two semantics-preserving runtime A/Bs followed:

| paired 40-inference run | p50 ms | p95 ms | achieved Hz | result |
|---|---:|---:|---:|---|
| ordinary `session.run()` split chain | 355.15 | 363.23 | 2.818 | baseline |
| device-resident conditioning/split chain | 351.70 | 354.57 | 2.846 | exact output |

The opt-in `--xvla-iobinding` path therefore reduced p50 by 1.0% and p95 by 2.4%.
It removes conditioning and hidden-state round trips, but the host still performs the
inter-step interpolation. All eight saved paired chunks were bit-identical.

A second candidate fuses the exact interpolation into `denoise_0`. Its graph inputs are
`x1`, previous `action`, `t`, `proprio`, and `cond_tokens`; with IOBinding, only the final
action crosses back to the host. The candidate bundle passes its full manifest and all
four denoiser graphs pass ONNX checker. On the real-IR fixture:

- CPU ORT FP32 has cosine minimum 1.000000 / 0.021% maximum range error versus Torch;
- fused versus ordinary CPU ORT differs by at most 4.172e-7;
- Jetson TensorRT FP16 has cosine minimum 0.9999791 / 0.422% versus Torch;
- two fresh TensorRT processes are bit-identical;
- a separate profile sees TensorRT node events in 12/12 graphs with no fallback.

The cold fused cache built in 380.30 seconds; its next identity-verified load took 26.25
seconds. The longer paired real-input result does not justify promotion:

| paired 40-inference real-IR run | p50 ms | p95 ms | achieved Hz | result |
|---|---:|---:|---:|---|
| partial IOBinding, host interpolation | 339.28 | 350.10 | 2.936 | baseline |
| full IOBinding, fused interpolation | 340.55 | 349.77 | 2.922 | parity PASS |

Full residency reduces Python/numpy time from 19.180 to 17.577 ms, but measured graph
time rises enough to offset it. The code and bundle option are retained as experimental;
the ordinary bundle remains the deployment default. Evidence includes
[`fused-trt-real-ir-parity.json`](fused-trt-real-ir-parity.json),
[`fused-long-ab-parity.json`](fused-long-ab-parity.json), and
[`placement-profile-fused/`](placement-profile-fused/).

The engine-cache manifest is a larger startup win. Migrating the existing cache once
took 57.66 seconds total, including 36.80 seconds of subprocess validation. The next
fresh process verified the identity manifest in 5.04 seconds and loaded in 25.64
seconds total, down 59% from the earlier 62.53-second warm-cache process. Its eight
saved outputs were also bit-identical to the original TensorRT result.

A separate, non-headline validation inference enabled ORT profiling on every session.
All 12 expected graphs emitted TensorRT node events; none emitted CUDA or CPU fallback
node events. The compressed raw traces and parsed per-graph provider/time summary are in
[`placement-profile/`](placement-profile/). Its 609.07 ms profiled wall time includes
trace overhead and is not comparable to the latency table.

Caveats:

- The new gate replays recorded excavator IR frames/state; it does not exercise live
  camera capture or the controller, and it does not establish task quality.
- Model-only TensorRT RSS was about 6.22 GB. System RAM had only about 567 MB free and
  1.125 GB of swap was already used, so integrated camera/control headroom remains open.
- Full bundle hashing still costs about five seconds on a cache hit; this is intentional
  integrity checking, not engine construction.
- An integrated camera/control dry-run memory measurement and a trained candidate are
  still required before deployment.
