# EVO1 on Orin Nano: FP16 cache boundary and an FP32-accumulating output head

Two changes, plus the shared preprocessing rewrite (`bench/vendor/imaging.py`).

- **FP16 key/value cache.** `action_context` computes each layer's K/V once per
  observation; they crossed into `action_step` as FP32 and were cast back to FP16 on
  every one of the 32 steps (16 tensors of 577×896: ~1.6 GB of casts per inference).
  `export.sh` now keeps them FP16 (`fp16_mixed --half-io`). The ONNX arithmetic is the
  same (the producers were FP16); TensorRT picks somewhat different action-step
  tactics, so the actions are not bit-identical.
- **Output head.** `action_output`'s pool layer is a GEMV, `[1,44800] @ W[896,44800]^T`,
  that reads 80 MB per step. TensorRT runs it on an FP16-accumulating tensor-core
  tactic (`h16816gemm`), summing 44,800 products in FP16. `build_output.py --mode gemv`
  replaces it with a Triton GEMV (`kernels.py`) that accumulates in FP32 at the same
  ~100 GB/s. Standalone, its error against an FP64 reference is 0.029 % vs 0.84 %.

| 60-second policy, 2 views | p50 ms | denoise ms | process RSS MB | full chunk vs stock FP32 |
|---|---:|---:|---:|---|
| published engines, new preprocessing | 346.19 | 125.0 | 2454 | 0.063 % |
| FP16 K/V cache | 339.84 | 119.0 | 2334 | 0.075 % |
| + FP32-accumulating output head | 340.69 | 119.6 | 2284 | **0.034 %** (mean abs 0.00088, was 0.00160) |

The output head costs ~0.8 ms and halves the policy's action error: the clearest sign
so far that TensorRT's FP16-accumulating tactics cost accuracy (see the playbook).

Not kept: a Triton kernel for the vision attention (1025 tokens, 16 heads, 2 views)
reaches 1.39 ms per call against TensorRT's 1.49 ms — about 2 ms per inference.

## Reproduce

```bash
export/export.sh zuoxingdong/evo1_libero ~/bundles/evo1-libero-split-kv16   # fine-tuning machine
# on the Nano: a plain cache of the bundle, then the output head
.venv-ort/bin/python -m bench trt-split --model evo1-libero --bundle ~/bundles/evo1-libero-split-kv16 \
  --cache-dir ~/.cache/jetson-orin-nano-vla/evo1-kv16-trt --chain graph --duration-s 20 --label plain
cd experiments/evo1_triton
../../.venv-torch-xvla/bin/python build_output.py --mode gemv --bundle ~/bundles/evo1-libero-split-kv16 \
  --base-cache ~/.cache/jetson-orin-nano-vla/evo1-kv16-trt --out ~/.cache/jetson-orin-nano-vla/evo1-out-gemv
cd ../..
.venv-ort/bin/python experiments/evo1_triton/run_candidate.py trt-split --model evo1-libero \
  --bundle ~/bundles/evo1-libero-split-kv16 --cache-dir ~/.cache/jetson-orin-nano-vla/evo1-out-gemv \
  --chain graph --duration-s 300 --label evo1-new --out results/evo1-new.json
```

`bench_gemv.py` and `bench_attention.py` are the standalone kernel checks.
