# π0.5 action-expert attention on Orin Nano

π0.5's runtime builds one weight-stripped, refittable plan per layer kind and refits a
copy per layer. `build_templates.py --mode attention` rebuilds the **action** template
with one AOT Triton plugin (`kernels.py: mqa_attention_aot`) in place of its exported
QK / scale / mask / softmax / PV chain; the plugin has no weights, so refitting works
unchanged and every other template is copied byte-identical.

The action expert's attention is multi-query: 8 query heads × 10 action rows against
one K/V head of 531 keys (521 prefix + 10), D=256. TensorRT ran it 180 times per
inference at 0.12 ms on an 8-block grid. The kernel follows the exported arithmetic:
HALF QK^T with FP32 accumulation, the exact ×1/16, FP32 mask + softmax, normalized
probabilities rounded to HALF, HALF PV with FP32 accumulation — in two passes over the
keys, so the probabilities are rounded after normalization as in the graph. 99.6 % of
its outputs are bit-identical to an oracle of the exported graph, the rest one HALF ulp
away. 0.070 ms per call. Tiles must stay within 48 KiB of shared memory: TensorRT's AOT
launcher does not opt in to more, and the 78 KiB tile failed at enqueue.

| 300-second policy, compact prefix | p50 ms | p99 ms | rate | denoise ms | process RSS MB | energy |
|---|---:|---:|---:|---:|---:|---:|
| templates as exported | 447.64 | 448.40 | 2.23 Hz | 99.7 | 6039 | 10.27 J |
| Triton action attention | **440.13** | **440.95** | **2.27 Hz** | 91.7 | 5998 | 10.13 J |

The fixtures pass with the same worst error against the mixed-precision reference
(0.402 % of range); against the FP16 reference the worst full chunk moves 0.116 → 0.127 %.

Tried and not kept ([microbenchmarks](../../results/pi05-triton-20261010/)):

- **Prefill attention** (521 rows, the language template): the exact two-pass kernel
  takes 1.10 ms and a one-pass online variant 1.17 ms against TensorRT's 1.0 ms; with
  D=256 the accumulator traffic, not QK^T, limits it.
- **Fused gated MLP** (gate and up GEMMs side by side, GELU-tanh × up in the epilogue,
  the exported HALF rounding after every op): 9.86 ms per layer against TensorRT's
  9.32 ms for the gate GEMM with fused GELU, the up GEMM and the separate multiply.

## Reproduce on the Nano

```bash
cd experiments/pi05_triton
../../.venv-torch-xvla/bin/python build_templates.py --mode attention \
  --out ~/.cache/jetson-orin-nano-vla/pi05-attention-new
cd ../..
.venv-ort/bin/python experiments/pi05_triton/run_candidate.py trt-split --model pi05-libero \
  --bundle ~/bundles/pi05-libero-compact-split --cache-dir ~/.cache/jetson-orin-nano-vla/pi05-attention-new \
  --chain graph --duration-s 300 --label pi05-new --out results/pi05-new.json
```

`--base-cache` defaults to the plain `pi05-libero-compact-trt` cache that a normal
`bench trt-split --model pi05-libero` run builds. `bench_attention.py` and
`bench_mlp.py` are the kernel checks.
