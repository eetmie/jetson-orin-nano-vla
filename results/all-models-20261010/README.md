# All-model round, 10 October 2026

Paired 300-second runs (`sustained/`), each model's control with the new exact host
preprocessing (`bench/vendor/imaging.py`, π0.5's tap-by-tap resize) against its final
configuration. [`summary-300s.json`](summary-300s.json) has latency, RSS, board RAM,
energy and fixture error per run.

| model | published p50 | control | final | final configuration |
|---|---:|---:|---:|---|
| EVO1 LIBERO | 360.33 ms | 346.56 | **340.72** | FP16 K/V cache (`export.sh`), FP32-accumulating output head ([experiment](../../experiments/evo1_triton/)) |
| GR00T N1.6 | 288.09 ms | 278.37 | **276.49** | `half_boundary` on `kv_*`, `mod_*`, `vl`, `state_features`; bit-identical |
| GR00T N1.7 | 288.11 ms | 282.20 | **279.67** | `half_boundary` on `kv_*`, `mod_*`, `state_features`; bit-identical |
| X-VLA | 383.68 ms | 365.91 ([earlier run](../xvla-triton-20261010T1410Z/)) | **349.31** | denoise attention, last block, `half_boundary` on chunk hiddens |
| π0.5 LIBERO compact | 473.96 ms | — | **447.64** | preprocessing only |

EVO1's fixture error halves (0.063 → 0.034 % of range) with the FP32-accumulating
output head. `traces/` holds the graph-chain Nsight traces the round started from;
`evo1-60s/` and `groot-60s/` the screening runs; `microbench/` the EVO1 attention and
GEMV kernel checks.

## FP32 accumulation (now the default)

[`fp32-accumulate/`](fp32-accumulate/): 60-second screens and paired 300-second runs of
the same final configurations built with `--accumulate fp32`
([`summary-300s.json`](fp32-accumulate/summary-300s.json)).

| model | p50 above (TensorRT's choice) | p50 FP32 accumulation | full chunk vs stock FP32 |
|---|---:|---:|---|
| SmolVLA | 102.85 ms | 105.11 | 0.195 → 0.135 % |
| X-VLA | 349.31 | 353.65 | 0.048 → 0.042 % |
| EVO1 | 340.72 | 351.42 | 0.034 → 0.021 % |
| GR00T N1.6 | 276.49 | 284.65 | 0.262 → 0.060 % |
| GR00T N1.7 | 279.67 | 286.80 | 0.062 → 0.041 % |

Engines and shared scratch keep their sizes and board RAM does not rise; process RSS
reads higher for four of the five.
