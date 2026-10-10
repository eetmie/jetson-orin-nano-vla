# Optimization playbook

What the optimization rounds on every model found, as a checklist to run through for
each new one. FP16 only; nothing here quantizes. Each item says how to spot it, what it was
worth where it applied, and whether it changes any number.

## The loop

1. Trace the default graph chain (`nsys` + `bench.tools.nsys_trace --summarize`), group
   kernels into GEMM / attention / layout-copy / norms, and look at the full demangled
   names and grid sizes (`CUPTI_ACTIVITY_KIND_KERNEL` joined to `StringIds`).
2. Read the exported ONNX and the TensorRT engine (`create_engine_inspector`, DETAILED
   verbosity gives the tactic per layer) before writing a kernel.
3. Change one thing, into a new cache, with a manifest pinning graph and engine hashes.
4. Gate: the bundle fixture (full chunk against stock FP32), plus a full-action stress
   check against the control over varied images, prompts, state and noise. Claim
   "bit-identical" only when `np.array_equal` says so.
5. Screen with 60-second windows, confirm with paired 300-second runs. Compare process
   RSS (`process.windows.load.rss_mb`), not just board RAM, which drifts with other
   processes. A gain that costs RAM on an 8 GB board has to clearly pay for it.

## Findings

| # | look for | fix | worth | numbers |
|---|---|---|---|---|
| 1 | `ScatterND` in the ONNX (PyTorch slice assignment, e.g. LeRobot's `apply_rope`) | rewrite with `torch.cat` at export | SmolVLA −7.5 ms (denoise 42.9→37.6, prefill 11.2→9.6), −100 MB board RAM | same arithmetic, bit-equal in PyTorch |
| 2 | attention masks built from constants only (no network input upstream) | don't read them in a custom kernel | SmolVLA vision mask: part of −6 ms | exact (adds zeros) |
| 3 | per-step work in a denoise loop that depends only on the observation | compute once per observation | SmolVLA cross-attention K/V −2.1 ms; GR00T AdaLN modulation + cross K/V earlier | same math, moved |
| 4 | the last block's rows that are sliced away before the decoder | compute only the decoded rows (keys/values keep all) | X-VLA last block −5.6 ms | row-wise ops, exact up to GEMM tactic |
| 5 | FP32 engine boundaries where the producer emits FP16 cast up and the consumer casts back down; find them with `python -m vla_common.boundary_report <bundle>` | FP16 at the boundary: `fp16_mixed --half-io` at export, or `python -m vla_common.half_boundary` on an existing bundle (refuses anything not exact) | EVO1 K/V cache −6.4 ms, −120 MB RSS; GR00T `kv_*`/`mod_*` −2.5 ms each; X-VLA chunk hiddens −1.9 ms, −38 MB | same ONNX arithmetic; GR00T bit-identical, others within TensorRT tactic changes |
| 6 | host preprocessing: numpy `/255`, normalize, transposes, numpy resizes, cameras one after another | `cv2.LUT` through a float table built by the model's own expression (`bench/vendor/imaging.py`), the cameras on a thread pool, or upload uint8 and gather on the GPU | SmolVLA 3.9→0.5 ms (GPU); X-VLA 20.2→2.6; EVO1 20.8→7.7; GR00T N1.6 14.2→5.2, N1.7 9.4→3.2; π0.5 28.5→9.7 (also a tap-by-tap resize) | bit-identical (tests + board check) |
| 7 | TensorRT's own MHA on short or odd sequences, or a plugin fed through transposes / `__myl_Resh` copies | Triton flash attention reading the projection's own layout (flat `[1,S,H*D]` or fused QKV) | SmolVLA vision 0.71→0.48 ms/call; X-VLA denoise 0.16→0.10 ms/call | SmolVLA bit-identical to its previous kernel; X-VLA within 0.04 % |
| 8 | a Triton tile chosen before a compile-time change | re-sweep after every constexpr change | SmolVLA 64×64→128×64: −2.6 ms | bit-identical |
| 9 | conv patch embeddings on `sm75 implicit_gemm` / `sm50 conv2d` kernels | patchify + MatMul at export | SmolVLA −0.8 ms | FP32-rounding level |
| 10 | builder optimization level | levels 3–5 | ≤1 % per engine, often noise; SmolVLA prefill got slower at 3 | — |
| 11 | a bandwidth-bound GEMV/GEMM on an FP16-accumulating tactic (`h16816gemm` with a small grid, huge K) | Triton GEMV accumulating in FP32 | EVO1 output head (K=44800): same speed, policy error 0.075 % → 0.034 % | better accuracy, not exact |
| 12 | a conv run in NCHW between transposes to and from tokens (DaViT positional convs on `sm50 conv2d` kernels, `__myl_*Tran*` copies around them) | a token-layout (channels-last) depthwise conv + residual plugin | X-VLA vision 92.2 → 81.1 ms, −109 MB RSS | exact against an FP32 conv; within 0.04 % end to end |

A Triton tile must also fit the 48 KiB of shared memory a kernel gets without an
opt-in: TensorRT's AOT launcher does not opt in, and a 78 KiB π0.5 tile failed at
enqueue (`plugin.py` asserts it now).

Rejected on RAM: feeding SmolVLA's attention plugin one fused QKV projection (−0.38 ms
for +9.3 MB RSS).

## Precision to keep in mind

- TensorRT picks FP16-accumulating GEMM tactics for many layers (`h16816gemm`,
  `xmma_gemm_f16f16_f16f16_f16`), and 10.16 has no builder flag against it. Custom
  kernels here accumulate in FP32. Worst spot seen: SmolVLA's K=12288 connector.
- That choice buys speed: on the board, cuBLAS FP16-accumulate GEMMs run ~25 % faster
  than FP32-accumulate at these sizes (X-VLA fc1 262×4096×1024: 7.3 vs 5.8 TFLOPS;
  1024×3072×768: 9.5 vs 7.5). Forcing FP32 accumulation costs roughly that on the
  affected GEMMs. Dense FP32-accumulate GEMMs top out near 7.5 TFLOPS here, and Triton
  tiles only tie TensorRT on X-VLA's M=262 GEMMs, so those are not a speed target.
- Measured cost of that in accuracy: replacing one FP16-accumulating GEMV (EVO1's
  K=44800 output head) with an FP32-accumulating one halved the whole policy's action
  error against stock FP32.
- π0.5's exporter already has the general fix: every MatMul is written as
  Cast(FP32) → MatMul → Cast(FP16). The operands are FP16 values, so the products are
  exact, and TensorRT turns the pattern into FP16-input / FP32-accumulate tensor-core
  kernels (`f16f16_f16f32`, `s16816gemm`). **This is now the default for every engine**
  (`groot_trt.accumulate_fp32`, applied at build time; `bench trt-split --accumulate
  auto` and the experiment builders' `--accumulate auto` turn it off, and the two are
  cached separately). It also clears TF32, so FP32 graphs run in real FP32.
  `python -m vla_common.fp32_accumulate` does the same to an ONNX bundle.

  | model (60 s) | p50, TensorRT's choice → FP32 | cost | full-chunk max error | mean error |
  |---|---|---:|---|---:|
  | SmolVLA | 102.78 → 105.16 ms | +2.3 % | 0.195 → 0.135 % | −31 % |
  | X-VLA | 348.6 → 353.0 ms | +1.3 % | 0.048 → 0.042 % | ≈ |
  | EVO1 | 340.7 → 351.4 ms | +3.1 % | 0.034 → 0.021 % | −47 % |
  | GR00T N1.6 | 276.5 → 283.5 ms | +2.5 % | 0.262 → 0.060 % | −46 % |
  | GR00T N1.7 | 279.7 → 285.1 ms | +1.9 % | 0.062 → 0.041 % | −45 % |

  X-VLA's GEMMs were already mostly FP32-accumulating, hence the small change there.

- `groot_trt.build_one` leaves TF32 enabled, so FP32 engines (projectors) run TF32;
  `pi05_trt` clears it.

## Per model

| model | done | still open |
|---|---|---|
| SmolVLA | 126.8 → 102.9 ms ([results](../results/smolvla-native-20261010T1247Z/summary.json)); FP32 accumulation by default | — |
| X-VLA | 383.7 → 342.1 ms: denoise attention, last block, LUT preprocessing, FP16 boundaries, token-layout depthwise convs in the vision tower ([experiment](../experiments/xvla_triton/)) | the six stage-entrance convs (~1 ms). Window attention pads 14×14 to 24×24 with zero keys: model semantics, keep |
| EVO1 | 360.3 → ~340 ms: preprocessing, FP16 K/V cache, FP32-accumulating output head ([experiment](../experiments/evo1_triton/)) | GEMM-bound vision (TensorRT's 1025-token MHA is already within 7 % of a Triton kernel) |
| GR00T N1.6 / N1.7 | 288 → ~275 / ~278 ms: preprocessing, FP16 `kv_*`/`mod_*` boundaries (bit-identical) | GEMM-bound; N1.7 `ds_*` boundaries need the deepstack scatter to work in FP16 bytes |
| π0.5 | 474 → 440.1 ms: preprocessing, Triton action-expert attention ([experiment](../experiments/pi05_triton/)) | prefill attention and the gated MLP: Triton kernels do not beat TensorRT there (D=256 attention 1.10 vs 1.0 ms; fused gate/up/GELU 9.86 vs 9.32 ms per layer) |
