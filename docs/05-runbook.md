# 5. Runbook — the bench day, in order

Ordered so the cheapest failures happen first. Budget ~2 h per model including the cold
TensorRT build; the measuring itself is minutes.

## 0. Artefacts

```bash
scripts/fetch_models.sh smolvla-base     # weights + the public split ONNX
scripts/fetch_models.sh xvla-base        # weights only — no split ONNX is published
```

Sizes: SmolVLA ~0.9 GB weights + 1.6 GB ONNX; X-VLA ~3.5 GB weights. Check the disk
before starting.

For a locally fine-tuned policy, rsync the `pretrained_model/` directory and the split
export instead, and sanity-check that they refer to the same checkpoint — the bundle's
`export_info.json` names the one it came from, and `PARITY.txt` records the split-vs-
torch check run at export time. If those disagree, the benchmark is comparing two
different models and every parity number is meaningless. Copy `pretrained_model/` only;
`training_state/` is another 400 MB of optimizer state inference never touches.

X-VLA has no published ONNX. Export it on a machine with room (not this board) and
rsync the graphs over — see `docs/03-backends.md`.

## 1. Board state

```bash
scripts/00_host_prep.sh          # MAXN_SUPER + pinned clocks + cache dir
scripts/00_host_prep.sh --verify
```

Confirm swap exists — the stock 2 GB is enough for the split builds (measured
2026-08-25); 16 GB is only needed if you are attempting the monolith. Close the
browser and the editor — the 8 GB is shared.

## 2. Environments

```bash
scripts/10_env_torch.sh          # asserts torch.cuda.is_available()
scripts/13_env_torch_xvla.sh     # only if benchmarking X-VLA (lerobot 0.6.1)
scripts/11_env_ort.sh            # asserts TensorrtExecutionProvider registers
```

Each fails loudly rather than proceeding into a wrong measurement. If `10_env_torch.sh`
passes but CUDA is missing, stop: everything downstream is invalid.

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
M=smolvla-base
CKPT=~/bundles/$M-torch
BUNDLE=~/bundles/$M-split

.venv-torch/bin/python -m bench torch --model $M --checkpoint $CKPT \
    --weights float32 --autocast off --iters 30 --label $M.torch-fp32
```

Start at 30 iterations. If a run is ~1 s per inference, 100 iterations is five minutes
of waiting for a number you already have.

Then the FP16 variants (SmolVLA):

```bash
... --weights float32 --autocast float16                --label $M.torch-amp16
... --weights float16  --autocast off --patch-half-out  --label $M.torch-half16
```

Check parity immediately — before spending an hour on TensorRT:

```bash
.venv-torch/bin/python -m bench parity results --reference $M.torch-fp32
```

If `torch-half16` fails parity against `torch-fp32` on Orin, that is a headline finding
on its own (it passed on Blackwell at cosine 0.999999).

## 5. The split TensorRT path

**The first run builds every engine, one subprocess per graph** — around 5 min for
SmolVLA's nine, and longer for X-VLA's twelve (not yet timed on this board). Do not run anything else on the board while it builds: two
builds in one process OOM 8 GB, and a browser does the same from outside.

```bash
.venv-ort/bin/python -m bench ort-split --model $M --bundle $BUNDLE \
    --precision fp16 --iters 100 --label $M.ort-split
```

Check an ORT placement profile before trusting the configured provider priority. If a heavy graph runs on
`CUDAExecutionProvider` instead of `TensorrtExecutionProvider`, the engine did not build
and the number is not what it looks like.

Memory-tight build: `--drop-cuda-ep` frees the 3 GiB CUDA arena, and
`--trt-workspace-mb 512 --trt-opt-level 2` are the settings already proven to fit here.

Then the first item off the optimization backlog, which is one flag:

```bash
.venv-ort/bin/python -m bench ort-split --model $M --bundle $BUNDLE \
    --projectors gpu --label $M.ort-split.proj-gpu
```

See `docs/06-optimization-backlog.md` for what else is one flag away and what is not.

## 6. The monolith A/B

Worth doing before any third-party runtime, because it establishes what a monolithic
graph does on this board independently of anyone's tooling.

```bash
.venv-ort/bin/python -m bench ort-mono --model $M --onnx <mono>.onnx --bundle $BUNDLE \
    --label $M.mono-trt
.venv-ort/bin/python -m bench ort-mono --model $M --onnx <mono>.onnx --bundle $BUNDLE \
    --no-trt --label $M.mono-cuda-ep
```

Read `active_provider` and `trt_engine_cached` in the result before reading the latency.
A silent CPU fallback still returns a finite, plausible action chunk.

## 7. Sustained run, for thermals

Short runs cannot tell you whether latency holds. This is the run that answers it.

```bash
.venv-ort/bin/python -m bench ort-split --model $M --bundle $BUNDLE \
    --duration-s 300 --label $M.ort-split.sustained
```

Read `drift_q4_vs_q1_pct` and `tj max °C` together. A flat drift is a result too — it
would mean the board sustains this load at MAXN, which nothing here has established.

## 8. Report

```bash
.venv-torch/bin/python -m bench report results --out docs/RESULTS.md
```

Commit `results/*.json` alongside it. The JSONs are the evidence; the markdown is the
summary.

## Or just

```bash
MODEL=smolvla-base scripts/run_all.sh
MODEL=xvla-base    scripts/run_all.sh
```

which runs the supported matrix in order and keeps going past a failure. The monolith remains opt-in.

## If the actions matter, use real frames

Synthetic observations are fine for latency, power and CPU — the transformer does the
same work whatever the pixels are. They are not fine for judging what a runtime
*predicts*, because a procedural scene is out of distribution for any trained policy.

```bash
python -m bench.tools.extract_frames --video <dataset>/videos/.../episode_000000.mp4 \
    --out frames/ --count 30 --stride 10
... -m bench ort-split --model $M --bundle $BUNDLE --obs frames:frames/
```

Use the same `--obs` for every backend in a comparison, or the parity table is comparing
different inputs.
