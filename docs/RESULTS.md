# Results

_Not measured on the Orin yet. Run `python -m bench report results --out docs/RESULTS.md`
on the board and this file is overwritten with the real tables._

## What this is meant to replace

Everything below is either a prior measurement on this class of hardware or a published
claim. The point of the bench day is to turn every one of these rows into a measured
row, on one board, in one session, with the same observations.

### SmolVLA 450 M

| claim | source | note |
|---|---|---|
| split TRT: vision 33.1 ms, prefill 16.5 ms, decode 11.4 ms/step | measured on this board, 2026-06-17 | dummy inputs, per-engine |
| → projected ~170 ms @ 10 steps, ~110 ms @ 5 | derived from the above | end-to-end loop not yet measured |
| split TRT end to end, 1 camera, 10 steps | 210 ms/chunk, chunk 50, 4.8 Hz replan, 2.21 GB peak RSS | a later fine-tuned export |
| monolith on ORT CUDA-EP: 532 ms mean, 498 ms p50 | measured on this board | no TRT, no engine build |
| monolith TRT build on 8 GB | fails — OOM or Err 10 after up to 85 min | ~6 GB node-count-independent floor |
| "SmolVLA does 10 Hz on an Orin Nano" | `isaacsim_vla_ws` demo | that is the *observation* rate, not inference |
| ~25 ms FP16 on Orin Nano 8 GB | Tether `docs/HARDWARE.md` | an order of magnitude off the split-path measurements above |

### X-VLA 880 M

| claim | source |
|---|---|
| split TRT end to end, 1 camera, 10 steps: 390 ms/chunk, chunk 30, 2.56 Hz replan | measured on this board |
| peak RSS 5.71 GB — 1.47 GB free of 7.4 GB | measured on this board |
| 12 engines; ~6.9 bytes/param resident vs SmolVLA's ~4.4 | measured on this board |
| build peak ≈ 3.18 GB + 5.63 × (FP32 weight GB) | measured on this board |
| parity vs the PyTorch reference: cosine 1.000000 | measured on this board |

None of these were taken with a common harness, the same observations, or CPU/power
instrumentation — which is what this repo adds.

## Harness validation (2026-08-24, DGX Spark GB10 — not the target board)

The measurement apparatus was exercised end to end before the bench day, on the machine
that holds the checkpoints and exports. **These are not Orin numbers** — the Spark was
under other load and its ONNX Runtime has no TensorRT EP, so the split path ran on the
CPU provider. What they establish is that the comparison itself is sound.

Cross-backend parity, identical seeded observations and identical injected noise, all
paths going through the harness's own preprocessing and normalization:

| pair | cosine (min) | max abs diff | % of range |
|---|---|---|---|
| split ONNX vs `torch-fp32` | **1.0000000** | 1.40e-06 | 0.0002 |
| `torch-half16` vs `torch-fp32` | 0.9999990 | 1.18e-03 | 0.059 |
| `torch-amp16` vs `torch-fp32` | 0.9999931 | 3.24e-03 | 0.162 |

Two things follow. The harness feeds every backend the same thing and compares them
correctly — the split-vs-torch agreement matches the bundle's own `PARITY.txt` (cosine
1.0000000, max abs 2.6e-06) produced by a completely separate script. And a hard FP16
weight cast did **not** collapse this fine-tuned SmolVLA checkpoint on Blackwell, which
is the first evidence against the vision-tower overflow being fatal. It still has to be
re-checked on the Orin: different hardware, different kernels.

Backend wiring was confirmed too: the split path reports prefix 113 / 1 camera slot /
chunk 12 read off the graphs, ten decode calls and ten calls to each of the four
CPU-side projectors per inference, and a (12, 4) action chunk.

One early number for optimization backlog item 0, from that CPU-provider run — the
absolute values mean nothing, the ratio is the point: the four per-step projectors
summed to **16.5 ms** against **611 ms** of decode. If that ratio roughly holds on the
Orin, where decode is ~11 ms/step, the CPU-side projectors are worth the flag rather
than a rounding error. That is a guess until it is measured on the board, which is what
`docs/06-optimization-backlog.md` item 0 is for.
