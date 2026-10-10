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
