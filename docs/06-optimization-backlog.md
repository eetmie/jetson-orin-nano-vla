# 6. Optimization status

The deployable SmolVLA path is already narrowed to a small set of choices. Current
defaults are FP16, GPU projectors, IOBinding, and the export's full denoise-step count.
Each optimization is kept only when its output and resource impact are measured.

## Current audited result

Pinned Orin Nano, two real camera views, ten denoise steps:

| configuration | p50 ms | p95 ms | Hz | vision ms | CPU cores busy | result |
|---|---:|---:|---:|---:|---:|---|
| PyTorch FP32 | 1167.93 | 1176.65 | 0.856 | — | 0.70 | reference |
| split FP16 | 189.89 | 190.93 | 5.248 | 91.234 | 0.34 | unmodified export |
| vision NaN-guard candidate | **164.66** | **165.88** | **6.044** | **66.232** | 0.38 | experimental |
| candidate repeat | 164.85 | 166.28 | 6.052 | 66.326 | 0.39 | repeatable |

Full latency, footprint, power, environment, and action data are retained in
[`results/audit-patch-smolvla/`](../results/audit-patch-smolvla/).

## Kept improvements

### GPU projectors

The four per-step action/time projectors previously consumed substantial CPU time.
Moving them to the GPU returned CPU headroom and preserved outputs. TensorRT can decline
these tiny graphs; CUDA EP execution is expected and should not be described as TRT.

### IOBinding

The denoise loop used to resend 32 KV tensors—7.2 MB per step—from numpy. Binding the
cache once removes roughly 72 MB of host-to-device traffic per inference. The validated
A/B was bit-identical. IOBinding is now the default; `--no-iobinding` exists only for
regression comparisons.

### Vision NaN-guard removal

The optimized vision graph removes twelve redundant
`Softmax -> IsNaN -> Where -> MatMul` guards after deterministic CPU validation. It
cuts the measured vision stage from 91.234 to 66.232 ms.

It is still gated. Against the FP32 Torch reference:

| bundle | cosine min | max abs | % of action range | verdict |
|---|---:|---:|---:|---|
| unmodified FP16 | 0.9993390 | 0.3641 | 2.052% | FAIL |
| optimized candidate | 0.9989535 | 0.2780 | 1.566% | FAIL |

The candidate is closer by maximum error but misses the strict 0.999 cosine / 1% range
gate. Keep it experimental until robot validation or a better export resolves the
difference.

## Behaviour-changing option

Reducing denoise steps is faster, but it changes the policy rather than merely its
runtime:

| steps | historical p50 ms | cosine vs 10 steps | max error as range |
|---:|---:|---:|---:|
| 10 | 179.8 | — | — |
| 5 | 116.7 | 0.981 | 11.0% |
| 4 | 103.9 | 0.960 | 15.0% |

Do not select a lower step count from latency alone. Validate action-to-motion
correlation and complete robot episodes with the fine-tuned checkpoint.

## Next work

1. Resolve or characterize the SmolVLA FP32 parity gap before promoting the vision
   candidate.
2. Complete the X-VLA audit with a runnable Torch reference and fresh Orin measurements.
3. If more SmolVLA latency is needed, evaluate fusing the four projectors into decode;
   it removes repeated launches without joining the large vision/prefill graphs.
4. Re-run the current `kaivuriprokkis` checkpoint after robot-side updates.

Higher TensorRT builder levels were dropped after failing within the memory budget.
Precomputing the time embedding has only a roughly 2 ms ceiling after GPU projectors.
A whole-policy engine was dropped because its build exceeds 8 GB. Detailed chronology
and audit evidence live in [`07-audit-followups.md`](07-audit-followups.md), leaving
this file as the current decision surface.
