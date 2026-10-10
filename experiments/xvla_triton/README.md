# X-VLA denoiser attention on Orin Nano

X-VLA's denoiser runs all 262 tokens (30 action, 200 conditioning, 32 soft prompt)
through 24 bidirectional blocks on each of 10 steps, so nothing can be cached across
steps. Two changes here, plus one in the shared runtime:

- **Attention.** Every block's exported QK / FP32 softmax / PV chain becomes one AOT
  Triton plugin reading the fused `[1,262,3072]` QKV projection and writing what the
  output projection reads (`kernels.py`). It keeps the HALF 0.3535 scaling of Q and K
  and the HALF score rounding, with FP32 online softmax and accumulation. 0.10 ms per
  call in the policy against 0.16 ms for TensorRT's fused MHA.
- **Last block.** The graph decodes only the first 30 rows after the last block, and
  everything after attention is row-wise, so that block computes 30 query rows (keys
  and values keep all 262) and its projection, residual and MLP run on 30 rows.
- **Preprocessing** (`bench/vendor/xvla_trt.py`, all X-VLA runs): `cv2.LUT` through a
  float table reproduces the `/255` + ImageNet normalization bit for bit and replaces
  the numpy transposes: 20.2 → 2.6 ms for three 480×640 views.

- **FP16 boundaries.** The vision/denoise chunk hiddens, `vlm_features` and
  `cond_tokens` round-tripped FP16→FP32→FP16 between engines; `vla_common.half_boundary`
  passes them in FP16 (`xvla-base-split-h16`). Five-minute run: **349.31 ms**, 2.86 Hz,
  process RSS 2587 MB, full chunk 0.048 % of range.

`build_denoise.py` checks every link of the exported chain (index maps of every
reshape/transpose/split, the SDPA scale subgraph, the residual) before rewiring.

| 60-second policy, 3 views | p50 ms | rate | denoise ms | process RSS MB |
|---|---:|---:|---:|---:|
| rebuilt control | 381.98 | 2.62 Hz | 266.6 | 2746 |
| Triton attention | 373.51 | 2.68 Hz | | 2632 |
| + last block on action rows | 367.88 | 2.72 Hz | 252.6 | 2578 |
| + `cv2.LUT` preprocessing | 350.47 | 2.85 Hz | 252.5 | 2605 |

Paired five-minute runs, both with the new preprocessing:

| 300-second policy | p50 ms | p99 ms | rate | process RSS MB | energy/inference |
|---|---:|---:|---:|---:|---:|
| rebuilt control | 365.91 | 368.68 | 2.73 Hz | 2593 | 8.37 J |
| attention + last block | **350.65** | **351.80** | **2.85 Hz** | **2571** | **8.20 J** |

Full chunk against stock LeRobot FP32: 0.050 % of range (control 0.054 %). The stress
check over six image kinds with their own proprio and noise stays within 0.038 % of the
control. Not kept: builder optimization level 5 (−0.18 ms). Triton GEMM tiles only tie
TensorRT on the M=262 projections ([microbenchmarks](../../results/xvla-triton-20261010T1410Z/microbench/)).

## Vision tower: depthwise convs in token layout

The export keeps DaViT's stage tensors in NCHW: every block runs its 3×3 depthwise
positional conv + residual in NCHW (TensorRT: an sm50 kernel, ~0.17 ms each) and
transposes to tokens and back around it. `build_vision.py --mode dwconv` replaces each
conv whose input comes from tokens and whose output goes back to tokens (42 of 48; the
six at engine inputs and after strided convs stay) with `dwconv_tokens_aot`, which reads
and writes `[views, H·W, C]` directly: FP32 taps, conv rounded to HALF, HALF residual
add. Against an FP32 conv reference it is exact on all four stage shapes; 0.045 ms at
14×14×1024, where 36 of the convs are. Index maps of every shuffle on both sides are
checked before rewiring.

| 300-second policy, FP32 accumulation | p50 ms | p99 ms | rate | vision ms | process RSS MB | energy |
|---|---:|---:|---:|---:|---:|---:|
| previous | 353.65 | 355.12 | 2.83 Hz | 92.2 | 2708 | 8.21 J |
| token-layout depthwise convs | **342.08** | **343.25** | **2.92 Hz** | 81.1 | 2599 | 8.01 J |

Full chunk 0.045 % of range (0.042 % before, mean error unchanged); the stress check
stays within 0.039 % of the control ([runs](../../results/xvla-vision-20261010/)).
Plugin attributes do not work with this TensorRT and numpy (scalar ints go through
`int(array)`, NDArray annotations fail registration), so `plugin.py` registers one op
per stage shape.

**Accumulation.** The build scripts now default to `--accumulate fp32` (every FP16 MatMul accumulates in FP32, see the [playbook](../../docs/07-optimization-playbook.md)); the tables above were measured with TensorRT's own choice, which `--accumulate auto` reproduces.

## Reproduce on the Nano

For the FP16-boundary bundle, first on the export machine:
`cp -a ~/bundles/xvla-base-split ~/bundles/xvla-base-split-h16 && (cd export && python -m vla_common.half_boundary --bundle ~/bundles/xvla-base-split-h16 --names 'hidden_in|cond_tokens|vlm_features')`,
then pass `--bundle ~/bundles/xvla-base-split-h16` and a plain cache of it below.

```bash
stamp=$(date -u +%Y%m%dT%H%MZ); C=~/.cache/jetson-orin-nano-vla
# needs the plain prebuilt cache $C/xvla-base-trt (any normal trt-split run builds it)
cd experiments/xvla_triton
../../.venv-torch-xvla/bin/python build_denoise.py --mode actions --out $C/xvla-actions-$stamp
cd ../..
.venv-ort/bin/python experiments/xvla_triton/check_variants.py --control $C/xvla-base-trt \
  --candidate actions=$C/xvla-actions-$stamp --out results/xvla-variants-new.json
.venv-ort/bin/python experiments/xvla_triton/run_candidate.py trt-split --model xvla-base \
  --bundle ~/bundles/xvla-base-split --cache-dir $C/xvla-actions-$stamp --chain graph \
  --warmup 10 --idle-s 3 --duration-s 300 --label xvla-actions-new --out results/xvla-actions-new.json
```

For the vision engines on top of a denoiser cache:
`../../.venv-torch-xvla/bin/python build_vision.py --mode dwconv --bundle <bundle> --base-cache <cache> --out <new cache>`.

`--mode baseline` rebuilds the unchanged denoiser as a control, `--mode attention` skips
the last-block change. `profile_candidate.py` traces a verified cache under Nsight;
`bench_attention.py` is the standalone kernel check. [Runs, manifests, trace and
summary](../../results/xvla-triton-20261010T1410Z/summary.json).
