# Results

_Not measured yet. Run `python -m bench report results --out docs/RESULTS.md` on the
Orin Nano and this file is overwritten with the real tables._

Until then, the numbers this repo is testing against, all from prior work on the same
class of hardware:

| claim | source | note |
|---|---|---|
| SmolVLA ~25 ms FP16 on Orin Nano 8 GB | Tether `docs/HARDWARE.md` | the claim under test |
| split TRT: vision 33.1 ms, prefill 16.5 ms, decode 11.4 ms/step | measured on this board, 2026-06-17 | dummy inputs, per-engine |
| → projected ~170 ms @ 10 steps, ~110 ms @ 5 | derived from the above | end-to-end loop not yet measured |
| monolith on ORT CUDA-EP: 532 ms mean, 498 ms p50 | measured on this board | no TRT, no engine build |
| monolith TRT build on 8 GB | fails — OOM or Err 10 after up to 85 min | ~6 GB node-count-independent floor |
| PyTorch SmolVLA on Orin Nano "10 Hz" | `isaacsim_vla_ws` demo | that is the *observation* rate, not inference |

The point of the bench day is to replace every projected row above with a measured
one, on one board, in one session, with the same observations.

## Harness validation (2026-08-24, DGX Spark GB10 — not the target board)

The measurement apparatus was exercised end to end before the bench day, on the
machine that holds the checkpoint and the export. **These are not Orin numbers** — the
Spark was under other load and its ONNX Runtime has no TensorRT EP, so the split path
ran on the CPU provider. What they establish is that the comparison itself is sound.

Cross-backend parity, identical seeded observations and identical injected noise, all
four paths through the harness's own preprocessing and normalization:

| pair | cosine (min) | max abs diff | % full stick |
|---|---|---|---|
| split ONNX vs `torch-fp32` | **1.0000000** | 1.40e-06 | 0.000 |
| `torch-half16` vs `torch-fp32` | 0.9999990 | 1.18e-03 | 0.059 |
| `torch-amp16` vs `torch-fp32` | 0.9999931 | 3.24e-03 | 0.162 |

Two things follow. The harness feeds every backend the same thing and compares them
correctly — the split-vs-torch agreement matches the bundle's own `PARITY.txt`
(cosine 1.0000000, max abs 2.6e-06) reached by a completely separate script. And a
hard FP16 weight cast did **not** collapse this fine-tuned checkpoint on Blackwell,
which is the first evidence against the vision-tower overflow being fatal here. It
still has to be re-checked on the Orin: different hardware, different kernels.

The backend wiring was also confirmed: the split path reports prefix 113 / 1 camera
slot / chunk 12 read off the graphs, 10 decode calls and 10 calls to each of the four
CPU-side projectors per inference, and a (12, 4) action chunk.
