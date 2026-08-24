# 5. Runbook — the bench day, in order

Ordered so the cheapest failures happen first. Budget ~2 h including the cold
TensorRT build; the actual measuring is minutes.

## 0. Get the artefacts onto the board (do this before anything else)

Two things come from the DGX Spark:

| what | why | size |
|---|---|---|
| the **split export bundle** | `ort-split` runs it; every backend reads its `stats.json`, `tokenizer/` and `export_info.json` | ~1.5 GB |
| the **LeRobot checkpoint** (`pretrained_model/`) | `torch` loads it | ~0.9 GB |

```bash
rsync -avP <spark>:~/Desktop/smolvla-digging-clean-ir12-35k/ ~/bundles/smolvla-digging-clean-ir12-35k/
rsync -avP <spark>:~/spark-projects/smolvla-spark-finetune/outputs/digging_clean/clean_ir12/checkpoints/035000/pretrained_model/ \
           ~/bundles/clean_ir12-035000/pretrained_model/
```

Copy `pretrained_model/` only — `training_state/` is another 400 MB of optimizer state
that inference never touches, and disk is not free here.

Sanity-check the pair before trusting any comparison: the bundle's `export_info.json`
names the checkpoint it was exported from, and its `PARITY.txt` records the split-vs-
torch check that was run at export time (cosine 1.0000000, max abs 2.6e-06). If those
two do not refer to the same checkpoint, the benchmark is comparing two different
models and every parity number below is meaningless.

## 1. Board state

```bash
scripts/00_host_prep.sh          # MAXN_SUPER + pinned clocks + cache dir
scripts/00_host_prep.sh --verify
```

Confirm swap exists. Close the browser and the editor — the 8 GB is shared.

## 2. Environments

```bash
scripts/10_env_torch.sh          # asserts torch.cuda.is_available()
scripts/11_env_ort.sh            # asserts TensorrtExecutionProvider registers
scripts/12_env_tether.sh         # runs `tether doctor` — keep its output
```

Each script fails loudly rather than proceeding into a wrong measurement. If
`10_env_torch.sh` passes but CUDA is missing, stop: everything downstream is invalid.

## 3. Prove the instruments before trusting them

```bash
.venv-torch/bin/python -m bench selftest --seconds 4
```

Expect `sampler: TegrastatsMonitor` and non-empty `power_mw` / `gpu_pct`. If it says
`PsutilMonitor`, `tegrastats` is not on PATH and the power and GPU columns will be
blank for the whole session.

## 4. PyTorch baseline first

No export, no engine build — so if this fails, the problem is the environment, not the
backend.

```bash
BUNDLE=~/bundles/smolvla-digging-clean-ir12-35k
CKPT=~/bundles/clean_ir12-035000/pretrained_model

.venv-torch/bin/python -m bench torch --checkpoint $CKPT --bundle $BUNDLE \
    --weights float32 --autocast off --iters 30 --label torch-fp32
```

Start at 30 iterations. If `torch-fp32` is ~1 s per inference on this board, 100
iterations is five minutes of waiting for a number you already have.

Then the two FP16 variants:

```bash
... --weights float32 --autocast float16                  --label torch-amp16
... --weights float16  --autocast off --patch-half-out    --label torch-half16
```

Check parity immediately — before spending an hour on TensorRT:

```bash
.venv-torch/bin/python -m bench parity results
```

If `torch-half16` fails parity against `torch-fp32` on Orin, that is a headline
finding on its own (it passed on Blackwell at cosine 0.999999).

## 5. The split TensorRT path

**The first run builds three engines, one subprocess per graph, ~5 min.** Do not run
anything else on the board while it builds — two builds in one process OOM 8 GB, and a
browser will do the same from outside.

```bash
.venv-ort/bin/python -m bench ort-split --bundle $BUNDLE --precision fp16 \
    --iters 100 --label ort-split-fp16
```

Watch the first-run log for the provider each graph landed on. If `smolvlm_vision`
reports `CUDAExecutionProvider` instead of `TensorrtExecutionProvider`, the engine did
not build and the number is not the number you think it is — the run metadata records
`providers_per_graph` for exactly this reason.

If the build struggles for memory: `--drop-cuda-ep` (frees the 3 GiB CUDA arena), and
`TRT_WORKSPACE_MB=512 TRT_OPT_LEVEL=2` are the settings already proven to fit here.

## 6. Tether

Expect this one to be the fight. Give it room:

```bash
.venv-tether/bin/python -m bench tether --export-dir <tether export dir> \
    --bundle $BUNDLE --startup-timeout 1800 \
    --providers TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider \
    --iters 100 --label tether-trt
```

If `/act` refuses every payload shape:

```bash
.venv-tether/bin/python -m bench tether-probe --url http://127.0.0.1:8000
```

and pass the right shape with `--payload-template`.

If `tether export` cannot run on the board, export on the Spark and rsync the export
directory. Record that in `--notes` — a number whose artefact was built elsewhere
needs to say so.

**A failure here is a result.** The JSON records it, the report prints it, and "the
monolithic path does not build in 8 GB" is exactly the sort of thing this repo is for.

## 7. Sustained run, for thermals

The short runs will not catch throttling.

```bash
.venv-ort/bin/python -m bench ort-split --bundle $BUNDLE --duration-s 300 \
    --label ort-split-fp16-sustained
```

Read `drift_q4_vs_q1_pct` and `tj max °C` together.

## 8. Report

```bash
.venv-torch/bin/python -m bench report results --out docs/RESULTS.md
```

Commit `results/*.json` alongside it. The JSONs are the evidence; the markdown is the
summary.

## Or just

```bash
BUNDLE=~/bundles/smolvla-digging-clean-ir12-35k \
CKPT=~/bundles/clean_ir12-035000/pretrained_model \
scripts/run_all.sh
```

which does steps 4–8 in that order and keeps going past a failure.

## If the actions matter, use real frames

Synthetic observations are fine for latency, power and CPU — the transformer does the
same work whatever the pixels are. They are not fine for judging what a runtime
*predicts*, because a procedural scene is out of distribution for a policy trained on a
sandbox.

```bash
python -m bench.tools.extract_frames --video <dataset>/videos/.../cam1/episode_000000.mp4 \
    --out frames/ --count 30 --stride 10
... -m bench ort-split --bundle $BUNDLE --obs frames:frames/ ...
```

Use the same `--obs` for every backend in a comparison, or the parity table is
comparing different inputs.
