# GR00T N1.6 / N1.7 Triton AOT attention

Attention replacements for the three-camera GR00T bundles on Orin Nano Super SM87,
JetPack 7.2.1 / TensorRT 10.16.2.10 / Triton 3.6.0. Engines build serially in the
compiler environment; inference uses the TensorRT runtime alone (no Torch, no Triton).

## Selected recipe: `--mode all`

| model | native p50 ms | Triton p50 ms | board RAM |
|---|---:|---:|---:|
| N1.6 | 278.92 | **272.49** | +47 MB |
| N1.7 | 277.81 | **272.39** | +58 MB |

300 s paired runs on the gathered-K/V bundles; evidence in
[`results/groot-triton-20261012/`](../../results/groot-triton-20261012/). Against seven
stock FP32 references the policy error is lower than native on both models.
`bench.tools.vla_setup` selects this mode for both GR00Ts.

| graph | kernel | replaces |
|---|---|---|
| `vision_*` | `attention72_aot` (N1.6, D = 72 as 64 + 16), `attention_aot` (N1.7) | SigLIP2 / Qwen3-VL ViT attention |
| `llm_*` | `causal_aot` | causal GQA self-attention; skips the masked key tiles |
| `dit_*` | `attention_aot` | DiT self-attention and cross-attention on gathered or masked K/V |

Modes: `vision`, `dit-cross`, `dit-all`, `llm`, `dit-llm`, `all` (all three), and
`baseline` (no replacement, same builder settings).

## Precision

FP16 weights and activations, FP32 accumulation everywhere. Vision and LLM kernels
compute Q·Kᵀ·scale in FP32 (`SCORE_FP32`). The exported graphs round the score to FP16
twice; reproducing that order made the LLM kernel less accurate than TensorRT's fused
attention (mean error +5..17 %), and is why the first vision kernels failed their wider
checks. DiT kernels keep the graph order, which measured at or below native. The LLM
kernel and short (≤64-key) DiT cross-attention use the two-pass order (normalize, then
round probabilities to FP16, as the graph does); the rest use online softmax.

## Matching and safety

`build.py` rewrites a chain only after verifying: static scales; the causal mask's
values (`triu(−1e4)`); Q/K/V index maps through every reshape, transpose, slice and
expand back to the producer tensor, with head/row strides fitted and checked; FP16-only
probability casts; the output layout; and each kernel's shared memory ≤48 KiB (the AOT
launcher does not opt in to more). Anything else fails the build. Gathered DiT
cross-attention (no mask, `kv_*` input) is recognized by its K/V source; its row counts
come from the graph, so any baked prompt works. `candidate.json` pins ONNX, engine,
auxiliary-input and timestep-constant bytes; `run_candidate.py` refuses a changed cache.

## Build and run

```bash
key=groot-n16-base                   # or groot-n17-base
bundle="$HOME/vla-bundles/$key"
work="$HOME/vla-runs/groot-triton"
.venv-triton/bin/python experiments/groot_triton/build.py \
  --bundle "$bundle" --base-cache "$work/native" --mode all --out "$work/all"
.venv-trt/bin/python experiments/groot_triton/run_candidate.py trt-split \
  --model "$key" --bundle "$bundle" --cache-dir "$work/all" \
  --chain graph --views 3 --accumulate fp32 --warmup 10 --duration-s 300 --out "$work/all.json"
```

`--base-cache` is an ordinary native cache of the same bundle (`bench trt-split` builds
one). `--graphs dit_0,llm_0` builds a partial cache; `--graph G --check-only` validates
the matching of one graph without building.

## Checks

- `bench_attention.py [--set gathered]`, `bench_attention72.py [--score-fp32]`,
  `bench_causal.py`: tile sweeps against Torch oracles of the graph's precision order
  (run with a Torch + Triton venv; the board's `.venv-torch` works).
- `stress.py`: 12 full action chunks against a matching native control (drift, not
  accuracy).
- Accuracy against stock FP32: `results/groot-kvg-20261012/tools/stockcheck.py` with
  extra references from `export/groot/reference*.py --seed N`.
- `profile.py`: Nsight trace of a verified candidate cache.

History: the first round (DiT only; vision rejected) is in
[`results/groot-triton-20261011/`](../../results/groot-triton-20261011/). Its `--mode
all` meant vision + DiT; it now also includes the LLM.
