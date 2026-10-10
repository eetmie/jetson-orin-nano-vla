# Initial SmolVLA Triton tests on Orin Nano

The Desktop triton-kernel-lab softmax examples compile and run on the actual 8 GB
Orin Nano Super (SM87). They also work inside serialized TensorRT engines through
the AOT plugin API. **Neither initial policy candidate improves on TensorRT.**
Keep the normal backend as the baseline; this directory is an opt-in experiment.

## Measured policy results

Two views, ten denoising steps, the existing graph chain, ten warmup calls, and one
60-second synthetic timing window per engine. All other engines are byte-identical
to the original cache. Control rebuilds the original vision graph using the same
builder settings and original timing cache as the candidates.

| vision engine | p50 ms | p95 ms | achieved Hz | calls | vision layers |
|---|---:|---:|---:|---:|---:|
| original graph rebuilt as control | 148.65 | 149.79 | 6.7189 | 404 | 116 |
| Triton softmax and surrounding casts | 170.95 | 172.13 | 5.8424 | 351 | 212 |
| Triton mask addition, casts, softmax and NaN guard | 151.56 | 152.34 | 6.5916 | 396 | 145 |

Both candidates pass the existing native full-chunk fixture gate and comparisons
of eight saved action chunks against the control, with identical observations and
injected noise. Maximum error across saved chunks is 0.108% and 0.147% of the
control's observed action range, respectively. These are numerical checks, not
robot task evaluations.

TensorRT's control already fuses mask handling, casts, softmax and the NaN guard.
Its inspected layer names include `AddCastMaxrSubExpSumDivMulCastIsnaSele`. The
narrow replacement breaks that fusion and introduces many more engine layers.
The broader replacement recovers most of the loss but still increases p50 by
1.96%. Engine layer counts support the fusion-boundary explanation; they are not
a measurement of each layer's time.

The corrected standalone sweep uses the same tensor for each variant. At shape
`[1,12,1024,1024]`, four-warp row softmax takes 1.031 ms versus 1.313 ms for the
FP32 TensorRT operator. FP16 input/output with FP32 arithmetic takes 0.519 ms
versus 2.865 ms for a standalone TensorRT Cast/Softmax/Cast graph. **That 5.52x
operator result does not describe production policy speed:** the production
engine fuses a wider sequence. The fused masked kernel takes about 0.778 ms;
one/two/four warps are effectively tied, while eight warps are slower.

## Integration and precision

The vision graph contains twelve attention softmax operations, each with 12 heads
and 1,024 keys. `build_vision.py` validates the exact exported chain before
rewiring it, and uses an isolated engine cache. `fp32` replaces Softmax only;
`fp16` also replaces its casts; `masked` includes the FP16 mask addition and
NaN-to-zero selection. The FP32-only integration mode is available but has not
been timed as a complete policy.

All softmax math stays FP32. The fused candidate rounds logits-plus-mask to FP16
before softmax, matching the export, then rounds probabilities to FP16 before
the NaN guard. Checks cover random inputs, partial and fully masked rows, large
logits with FP16 addition rounding, and infinities. Maximum operator error against
the numerical oracle is 1.526e-5; every checked output is finite.

Builds use TensorRT's [AOT plugin API](https://github.com/NVIDIA/TensorRT/blob/main/samples/python/quickly_deployable_plugins/README.md)
and bake the compiled kernel into the engine. Policy runs use `.venv-ort` with
**neither Torch nor Triton installed**, without importing the Python plugin.
The separate build/probe environment has Torch 2.11.0+cu130 and Triton 3.6.0;
the board has CUDA 13.2 and TensorRT 10.16.2.10. Torch prints an SM87 binary-support
warning, but the concrete CUDA and Triton probes pass. No packages were replaced.

TensorRT reports two DSL-injected hidden kernel arguments and pads them with
null/zero. The tested row kernels require no global scratch and pass the complete
policy tests; other Triton kernels need their own ABI/workspace validation.

The existing MAXN_SUPER and locked clocks were retained. OC3 event counters rose
in all three policy windows; these numbers reflect that power condition. The
monitor reported no active thermal cooling clamps. Peak whole-system RAM was
1,773 MB for control, 1,874 MB for narrow softmax and 1,882 MB for fused softmax.
These runtime samples exclude the build environment. Sixty seconds does not
establish five-minute endurance.

## Reproduce on the Nano

Run from `~/jetson-orin-nano-vla`. Use a new cache and result directory for each
build/run; the experimental builder and runner reject existing output engines
and measurement files. This example selects the fused candidate; use `baseline`
for the original-graph control or `fp16` for the narrow candidate.

```bash
smol_stamp=$(date -u +%Y%m%dT%H%M%SZ)
smol_cache="$HOME/.cache/jetson-orin-nano-vla/smolvla-triton-$smol_stamp"
smol_results="results/smolvla-triton-$smol_stamp"
mkdir -p "$smol_results"

.venv-torch-xvla/bin/python experiments/smolvla_triton/build_vision.py \
  --mode masked --out "$smol_cache" > "$smol_results/build.log" 2>&1

.venv-ort/bin/python experiments/smolvla_triton/run_candidate.py trt-split \
  --model smolvla-base --bundle ~/bundles/smolvla-base-split \
  --cache-dir "$smol_cache" --chain graph --views 2 \
  --warmup 10 --idle-s 3 --duration-s 60 \
  --label "smolvla-base.triton-$smol_stamp" \
  --out "$smol_results/policy.json" > "$smol_results/policy.log" 2>&1

.venv-torch-xvla/bin/python experiments/smolvla_triton/check_masked.py \
  --out "$smol_results/masked-correctness.json"
.venv-torch-xvla/bin/python experiments/smolvla_triton/bench_softmax.py \
  --out "$smol_results/softmax-bench.json"
```

Run GPU experiments sequentially. The normal CLI at an experimental cache would
rebuild from the original ONNX; only the experimental wrapper verifies and uses
the candidate manifest. Builds assume the existing SmolVLA bundle/cache paths.

## Evidence and next target

[Raw results and summary](../../results/smolvla-triton-20261010/) retain the three
policy runs, engine layer descriptions, engine/bundle hashes, manifests, logs
and source snapshots. `sources-fp16` and `sources-masked` match the kernel/plugin
hashes in the measured engine manifests. `sources-operator` captures the corrected
operator sweep. Earlier operator results remain under `archives`; use
`operator-rerun-20261010T0945Z` for the same-input sweep. Its preceding graph-capture
attempt failed on the legacy default stream; the corrected probe uses an explicit
stream and its successful result is retained separately.

Recreate the summary from the repository root with:

```bash
PYTHONPATH=. python3 results/smolvla-triton-20261010/summarize.py
```

The subsequent fused vision attention experiment is retained under
[`results/smolvla-attention-20261010T1007Z`](../../results/smolvla-attention-20261010T1007Z/).
Adding valid pointer/stride alignment information reduced the standalone AOT
kernel from 11.06 ms to about 0.71 ms. Its paired 300-second policy runs measured
128.27 ms p50 / 7.7879 Hz against 148.88 ms / 6.7095 Hz for the rebuilt original
vision control, with eight saved-action comparisons passing. The vision engine
uses FP32 accumulation and approximate online probability rounding; this remains
an optional numerical/performance experiment rather than a robot-quality result.

## Expert gated projections

The patched policy's new Nsight Systems graph trace takes 130.5 ms under profiling,
with 125.1 ms of kernel work. TensorRT already uses fused attention in all sixteen
expert blocks: the two attention kernel variants consume approximately 5.6 ms
per ten-step inference. Most residual/RMS normalization chains are also fused.
The paired feed-forward projections remain a larger target: about 11.9 ms across
160 calls in the graph trace. The separate per-layer TensorRT profiler confirms
the target but changes synchronization, so its totals are diagnostic only.

`ffn_kernel.py` tests two implementations of the 50×720 input projected into two
2048-wide branches. A direct dual-matmul kernel mostly loses to TensorRT. Packing
the gate/up weights in interleaved columns permits one larger Tensor Core matmul
followed by sigmoid, SiLU and gating in the same kernel. The selected tile is
64×128 output elements with K tiles of 32, four warps and two stages. The initial
same-input operator sweep measures 0.0746 ms versus 0.0829 ms for an isolated TRT
graph. Synthetic operator timings establish feasibility, not a full-policy gain;
the wider tile/pipeline sweep is retained separately.

`build_expert.py --mode ffn` matches and checks all sixteen exported gate chains,
packs their real HALF weights, and replaces the chains with AOT plugins. FP32
accumulation and HALF projection/sigmoid/SiLU/product rounding boundaries are
retained. `--mode baseline` rebuilds the unchanged expert using the same builder
settings. Both inherit identical improved vision and other engine bytes from
the verified base cache. Candidate manifests retain every engine digest and
the additional expert ONNX digest; runtime and stress wrappers verify these
before bypassing the normal prebuild. The compiled policy runs without Torch
or Triton installed in the measurement environment.

All twelve stress cases pass: fixture, black, grey, white, checkerboard and random
pixels, each with one/two cameras, different prompts and controlled state/noise.
Maximum full padded-action difference from the aligned control is 0.0503% of
range. Rebuilding the expert without the plugin produces identical actions.

| 60-second policy | p50 ms | p95 ms | denoise mean ms |
|---|---:|---:|---:|
| Previous aligned vision policy | 127.88 | 128.83 | 46.816 |
| Rebuilt expert control, same vision | 127.91 | 128.79 | 46.794 |
| Packed Triton expert, same vision | 126.82 | 127.73 | 45.722 |

The bounded runs show approximately 1.1 ms lower denoising latency with comparable
GPU clocks. [Raw runs, trace, operator sweeps and validation](../../results/smolvla-expert-20261010T1041Z/)
are retained with the [comparison summary](../../results/smolvla-expert-20261010T1041Z/summary.json).

The paired five-minute windows confirm a modest gain:

| 300-second policy | p50 ms | p95 ms | achieved Hz | denoise mean ms |
|---|---:|---:|---:|---:|
| Rebuilt expert control, aligned vision | 128.10 | 129.10 | 7.7968 | 46.852 |
| Packed Triton expert, aligned vision | 126.67 | 127.63 | 7.8834 | 45.716 |

Median latency drops 1.12%; candidate quartile means remain 126.71, 126.73,
126.75 and 126.66 ms. Eight saved complete-action comparisons pass with minimum
cosine 0.9999998 and maximum error 0.024% of reference range. Whole-board mean
RAM use is approximately 1.80 GB for the candidate versus 1.73 GB for control
(tegrastats reports MB). Average GPU clocks are 1009.48 versus 1008.94 MHz;
neither window reports active thermal cooling clamps. Existing OC3 counters
rise in both windows (61,685 versus 63,694), so this retains the board's existing
power condition rather than establishing performance without power limiting.

Serialized expert inspection finds sixteen AOT plugins, 48 GEMM layers versus
80 in control, and 356 total layer entries versus 374. All non-expert engines
retain identical bytes. Cache verification tests pass locally, with Python
assertions disabled, and in the Jetson measurement environment. The candidate
remains an explicit experimental cache; the ordinary backend uses its normal
ONNX engine build path.

## RAM and speed on the 8 GB Nano

The memory follow-up keeps the verified AOT engines and their precision unchanged.
`run_candidate.py --lean-tokenizer` loads the saved Rust tokenizer directly,
preserving newline handling, right padding and the export's **left truncation**.
It also bounds the prompt-embedding cache to eight recently used instructions
(approximately 1.41 MiB of embedding/mask arrays). The existing cache retained
every distinct task. `--trim-host-heap` returns freed glibc pages once after
initialization, outside graph capture and the inference loop.

The benchmark's environment recorder previously imported every optional framework
to read its version. `bench.runner.collect_env` now reads installed distribution
metadata and prefers the version of an already-loaded module. This prevents
unused framework imports from becoming part of the measured deployment RAM.
The three corrected fresh-process, two-camera, 60-second windows are:

| Configuration | p50 ms | p95 ms | Board RAM mean, MB | Process RSS mean, MB |
|---|---:|---:|---:|---:|
| Aligned vision, original expert, standard tokenizer | 127.97 | 128.89 | 1803.23 | 1494.2 |
| Aligned vision, original expert, lean runtime | 127.86 | 128.79 | 1759.05 | 1428.7 |
| Aligned vision, packed expert, lean runtime | 126.86 | 127.81 | 1790.62 | 1455.2 |

**Keep the packed expert plus lean runtime as the balanced experimental reference.**
It saves about 1 ms versus the lean original expert for approximately 27 MB more
process RSS and 32 MB more sampled board RAM. Both memory views are below the
standard-tokenizer original expert in this comparison. Retain the lean original
expert as the lower-RAM option. Board memory includes other processes and varies
between windows; process RSS and CUDA allocations overlap on this shared-memory
board and must not be added. Engine files retained on disk do not consume their
full size in runtime RAM simply by existing.

Standalone fresh-process probes recover 18–21 MiB of freed heap and retain
bit-identical full padded actions. Direct tokenizer comparisons pass for 215
instructions, including whitespace, newline, Unicode, special tokens and strings
longer than the 48-token contract. Token IDs, masks and scaled embedding rows
are exact, and evicted prompts reproduce their original embedding on reuse.
The native policy fixture gate remains enabled. The lean path is restricted to
the cached GPT2 tokenizer export used here; it is an optional experiment, not a
generic replacement for every Transformers tokenizer.

Reproduce the balanced configuration in the Torch-free runtime, using a new output:

```bash
.venv-ort/bin/python experiments/smolvla_triton/run_candidate.py trt-split \
  --model smolvla-base --bundle ~/bundles/smolvla-base-split \
  --cache-dir ~/.cache/jetson-orin-nano-vla/smolvla-expert-ffn-20261010T1041Z \
  --chain graph --views 2 --warmup 10 --idle-s 3 --duration-s 300 \
  --lean-tokenizer --trim-host-heap \
  --label smolvla-balanced-new --out results/smolvla-balanced-new.json
```

For the lower-RAM option, use the verified
`smolvla-expert-control-20261010T1041Z` cache with the same runtime flags.
[`probe_memory.py`](probe_memory.py) audits one candidate per fresh process;
[`check_tokenizer.py`](check_tokenizer.py) performs offline exact comparisons.
Results and both environment-recorder source versions are preserved in
[`results/smolvla-memory-20261010`](../../results/smolvla-memory-20261010/).

The selected configuration's final 300-second check completes 2,364 calls at
7.8768 Hz: p50 **126.78 ms**, p95 **127.71 ms**, and quartile means 126.94,
126.82, 126.78 and 126.73 ms. Whole-board RAM averages 1814.96 MB (p95 1828 MB,
peak 1984 MB); process RSS averages 1458.6 MB (peak 1479.8 MB). Eight saved
complete action chunks are bit-identical to the same expert engine before
runtime memory changes. No Torch, Triton or Transformers is imported in this
runtime. Shared engine scratch remains 20 MiB. These RAM figures include
existing system processes; the higher board-RAM average than the 60-second
window does not establish a policy-memory regression. GPU clocks average
1009.66 MHz and the monitor reports no active thermal cooling clamps; existing
OC3 events continue (63,066 during this window).

[`selection.json`](../../results/smolvla-memory-20261010/selection.json) records
the chosen engine cache, runtime flags, precision and lower-RAM fallback.

The largest remaining speed target was vision's feed-forward/matmul work: the
vision stage took approximately 64 ms for two cameras (see the next section). Inspect and measure
any wider GEMM/activation fusion against the existing TensorRT implementation.
Expert attention and most normalization/residual chains are already fused;
replacing their small operators separately is unlikely to be the best use of RAM
or development time. Weight quantization is a potential larger memory reduction,
but changes numerical behavior and needs [calibration and accuracy validation](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html), full-action checks and robot
task evaluation before adoption. Keep camera resolution and ten denoising steps
fixed in these comparisons so model workload remains comparable.

## Layout, cross-attention K/V and RoPE

Second round, from an audit of the selected configuration's trace and exported graphs.
Each row adds one change to the row above; 60-second two-view windows, same flags as
the selected run. Fixture is the full-chunk error against stock LeRobot FP32.

| change | p50 ms | rate | process RSS MB | fixture |
|---|---:|---:|---:|---:|
| selected configuration above | 126.87 | 7.87 Hz | 1484 | 0.166 % |
| vision attention on the flat projections, constant mask not read | 120.79 | 8.27 Hz | 1451 | 0.166 % |
| cross-attention K/V once per observation, FP16 KV cache | 118.70 | 8.41 Hz | 1458 | 0.177 % |
| uint8 canvas upload, SigLIP scaling on the GPU | 113.72 | 8.78 Hz | 1451 | 0.177 % |
| RoPE by concatenation instead of ScatterND | 106.26 | 9.40 Hz | 1415 | 0.177 % |
| attention tile 128×64 | 103.65 | 9.63 Hz | 1419 | 0.177 % |
| patch embedding as patchify + MatMul | **102.81** | **9.71 Hz** | 1413 | 0.195 % |

- **Vision attention.** The exported vision mask is built from constants only and is
  all-true for a full image, so adding it is exact; `attention_native_aot` does not read
  it, applies the HALF 0.3535 Q/K scale itself and reads Q/K/V and writes O in the
  projections' own layout. That removes four transposes per layer and, fed the
  `[1,1024,768]` projection outputs (`--mode triton-attention-flat`), three reshape
  copies too. Actions are bit-identical in all twelve stress cases.
- **Cross-attention K/V.** In the eight cross-attention layers the expert re-projects
  the fixed prefix K/V through its own `k_proj`/`v_proj` on every denoise step.
  `export.sh --hoist-cross-kv` emits those projections from prefill instead and keeps
  the KV cache FP16 between the engines (decode cast every input to FP16 anyway).
  PyTorch difference against the per-step projection: 0.0.
- **GPU image conversion.** 95 % of the CPU preprocessing was the float conversion.
  The device chain now uploads the uint8 canvas and a table gather does the scaling; a
  load-time gate checks it is bit-identical to the host conversion.
- **RoPE.** LeRobot's `apply_rope` writes its halves by slice assignment, which exports
  as 110 ScatterND nodes that TensorRT runs unfused on every Q and K. The concatenated
  form is bit-equal in PyTorch; denoising falls from 42.9 to 37.6 ms, prefill from 11.2
  to 9.6 ms.
- **Tile.** With the scale as a compile-time constant, 128×64 / 4 warps / 3 stages
  takes 0.477 ms per call against 0.58 ms for 64×64 (bit-identical).
- **Patch embedding.** The stride-16 conv ran on an sm75 implicit-GEMM kernel; the
  MatMul form matches it to FP32 rounding (1.5e-6). Both use FP16-accumulating kernels.

The paired five-minute check (both with the new GPU image conversion, so this isolates
the engine changes):

| 300-second policy | p50 ms | p95 ms | p99 ms | rate | process RSS MB | energy/inference |
|---|---:|---:|---:|---:|---:|---:|
| selected engines above | 121.54 | 122.57 | 122.97 | 8.22 Hz | 1460 | 2.54 J |
| this round | **102.85** | **103.54** | **103.90** | **9.70 Hz** | **1405** | **2.22 J** |

2,912 calls, quartile means 102.94 / 102.92 / 102.99 / 102.97 ms; GPU clocks average
1008.8 vs 1009.1 MHz, existing OC3 events in both windows, no thermal clamps.
Tried and not kept: decode at builder optimization level 5 (−0.08 ms) and one fused
QKV projection feeding the plugin (−0.38 ms for +9.3 MB RSS). An isolated trtexec
sweep found levels 3–5 worth ≤1 % per engine, and prefill slower at level 3.

Reproduce on the Nano (the bundle is exported on the fine-tuning machine):

```bash
export/export.sh lerobot/smolvla_base ~/bundles/smolvla-base-split-kv3 --views 2 --hoist-cross-kv
# on the Nano
stamp=$(date -u +%Y%m%dT%H%MZ); C=~/.cache/jetson-orin-nano-vla; B=~/bundles/smolvla-base-split-kv3
.venv-ort/bin/python -m bench trt-split --model smolvla-base --bundle $B \
  --cache-dir $C/smolvla-kv3-trt --chain graph --views 2 --duration-s 20 --label plain
.venv-torch-xvla/bin/python experiments/smolvla_triton/build_vision.py --mode triton-attention-flat \
  --bundle $B --base-cache $C/smolvla-kv3-trt --out $C/smolvla-kv3-flat-$stamp
.venv-torch-xvla/bin/python experiments/smolvla_triton/build_expert.py --mode ffn \
  --bundle $B --base-cache $C/smolvla-kv3-flat-$stamp --out $C/smolvla-kv3-ffn-$stamp
.venv-ort/bin/python experiments/smolvla_triton/run_candidate.py trt-split --model smolvla-base \
  --bundle $B --cache-dir $C/smolvla-kv3-ffn-$stamp --chain graph --views 2 \
  --warmup 10 --idle-s 3 --duration-s 300 --lean-tokenizer --trim-host-heap \
  --label smolvla-kv3-new --out results/smolvla-kv3-new.json
```

[Runs, stress checks, manifests, trace, microbenchmarks and the opt-level sweep](../../results/smolvla-native-20261010T1247Z/)
are summarized in its [`summary.json`](../../results/smolvla-native-20261010T1247Z/summary.json).

The new trace has the GPU busy 99 % of each call: 63 ms of GEMMs, 12.8 ms vision
attention, 11.9 ms expert gate/up, 6.8 ms fused MHA, 4.9 ms norms, 4.4 ms layout.
Many TensorRT GEMM tactics accumulate in FP16 (`h16816gemm`, `f16f16_f16f16_f16`),
including the vision QKV/fc2 projections and the K=12288 connector; TensorRT 10.16 has
no builder flag for it. That is the next quality question, not a speed one.

**Accumulation.** The build scripts now default to `--accumulate fp32` (every FP16 MatMul accumulates in FP32, see the [playbook](../../docs/07-optimization-playbook.md)); the tables above were measured with TensorRT's own choice, which `--accumulate auto` reproduces.

## Rebuild the expert engine

Build and measure on the Nano, from the repository root. Use unique names:

```bash
expert_stamp=$(date -u +%Y%m%dT%H%M%SZ)
expert_base="$HOME/.cache/jetson-orin-nano-vla/smolvla-triton-attention-aligned-20261010T1007Z"
expert_cache="$HOME/.cache/jetson-orin-nano-vla/smolvla-expert-$expert_stamp"
expert_results="results/smolvla-expert-$expert_stamp"
mkdir "$expert_results"
.venv-torch-xvla/bin/python experiments/smolvla_triton/build_expert.py \
  --mode ffn --base-cache "$expert_base" --out "$expert_cache"
.venv-ort/bin/python experiments/smolvla_triton/check_policy_variants.py \
  --control "$expert_base" --candidate "expert=$expert_cache" \
  --out "$expert_results/variants.json"
.venv-ort/bin/python experiments/smolvla_triton/run_candidate.py trt-split \
  --model smolvla-base --bundle ~/bundles/smolvla-base-split \
  --cache-dir "$expert_cache" --chain graph --views 2 \
  --warmup 10 --idle-s 3 --duration-s 300 \
  --label "smolvla-base.expert-$expert_stamp" --out "$expert_results/policy.json"
```

The trace wrapper accepts the same `--model`, `--bundle`, `--cache-dir` and
`--chain` arguments as `bench.tools.nsys_trace`; use `profile_candidate.py` under
Nsight to keep the verified experimental engines intact. `profile_expert_layers.py`
provides the separate diagnostic layer profiler. `bench_ffn.py` reproduces the
isolated sweep (`--extended` for the wider configurations).

## Credit

The row and online softmax examples come from Jeremy Gracey's
[triton-kernel-lab](https://github.com/JeremyGracey-AI/triton-kernel-lab).
The copied source and MIT notice are retained under `vendor/` (Copyright 2026
Jeremy Gracey). Our changes adapt the kernel ABI to TensorRT AOT and add the
model-specific mask/rounding/NaN handling. TensorRT provides the AOT plugin API.
