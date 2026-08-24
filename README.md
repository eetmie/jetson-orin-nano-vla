# jetson-orin-nano-vla

Benchmarking Vision-Language-Action (VLA) inference on an **8 GB NVIDIA Jetson Orin
Nano Super**, JetPack 7.2.

One model, one board, three runtimes, the same seeded observations fed to all of them:

| backend | what it is |
|---|---|
| **`torch`** | stock PyTorch — LeRobot `SmolVLAPolicy` straight off the checkpoint |
| **`ort-split`** | the split 9-graph ONNX export on ONNX Runtime + TensorRT EP (FP16) |
| **`tether`** | [FastCrest Tether](https://github.com/FastCrest/tether) `serve` + `/act` |

The model is SmolVLA 450M fine-tuned for excavator digging: chunk 12, 10 denoise
steps, one IR camera, 3-joint state, 4-dim joystick-rate action.

## What is measured, and why those things

The interesting question on this board is not "which is fastest". It is **what a
runtime leaves behind for the rest of the robot.** So every run reports:

- **Parity** — cosine and max-abs against a reference, on identical seeded noise, read
  as a fraction of full stick travel. The Orin is compute 8.7 and FP16 is the only
  fast precision it has; blanket FP16 on SmolVLA is exactly what collapsed the SigLIP
  vision tower to cosine 0.805 on Blackwell. A fast, wrong backend is worse than a
  slow one.
- **Latency** — p50/p95/max, plus per-quartile drift over a sustained run, because an
  Orin at MAXN throttles and a ten-second mean hides it.
- **CPU** — per-process cores consumed, idle vs. inferring. *This is the metric that
  decides the project.* The split path runs six of its nine graphs on the CPU EP and
  the whole denoise loop in numpy; "it runs on the GPU" is a claim about three graphs,
  not a pipeline. Cores spent here are cores the control stack cannot have.
- **RAM** — resident, peak, and whole-board. The 8 GB is unified, so the system figure
  is what decides whether something gets OOM-killed.
- **Power** — VDD_IN at MAXN_SUPER, integrated to **mJ per inference**, which is the
  only fair way to compare a fast-and-hungry backend against a slow-and-frugal one.

And the figure that actually bites in a chunked policy:

```
steps_consumed_per_inference = latency_s * fps
chunk_headroom_x             = chunk_size / steps_consumed
```

SmolVLA emits a chunk played open-loop while the next inference runs. Below 1× the
plan runs dry before its replacement lands. That is the threshold worth optimising
against — not a Hz number. (The widely quoted "SmolVLA does 10 Hz on an Orin Nano" is
an *observation* rate, not an inference rate.)

## Quickstart

```bash
scripts/00_host_prep.sh                 # MAXN_SUPER, pinned clocks, persistent engine cache
scripts/10_env_torch.sh                 # .venv-torch   (asserts torch sees the GPU)
scripts/11_env_ort.sh                   # .venv-ort     (asserts the TensorRT EP registers)
scripts/12_env_tether.sh                # .venv-tether

.venv-torch/bin/python -m bench selftest      # do the instruments read real numbers?

BUNDLE=~/bundles/smolvla-digging-clean-ir12-35k \
CKPT=~/bundles/clean_ir12-035000/pretrained_model \
scripts/run_all.sh                      # every backend, then docs/RESULTS.md
```

One backend at a time:

```bash
.venv-torch/bin/python -m bench torch --checkpoint $CKPT --bundle $BUNDLE \
    --weights float32 --autocast float16 --iters 100 --label torch-amp16
.venv-ort/bin/python   -m bench ort-split --bundle $BUNDLE --iters 100 --label ort-split-fp16
.venv-tether/bin/python -m bench tether --export-dir <export> --bundle $BUNDLE --label tether-trt

.venv-torch/bin/python -m bench parity results
.venv-torch/bin/python -m bench report results --out docs/RESULTS.md
```

## Layout

```
bench/
  runner.py        warmup -> idle baseline -> measured window -> one JSON per run
  monitor.py       tegrastats parser (RAM/CPU/GPU/temp/power); psutil fallback off-board
  procwatch.py     per-PID CPU + RSS from /proc, follows children (tether serves apart)
  obs.py           deterministic observations: same image, state, task AND noise draw
  parity.py        cross-backend action-chunk comparison
  report.py        result JSONs -> the markdown tables in docs/RESULTS.md
  backends/        torch_lerobot.py · ort_split.py · tether_http.py
  vendor/          smolvla_split.py, copied from kaivuriprokkis so the measured code is pinned
  tools/           extract_frames.py — real frames out of a LeRobot dataset
docs/              01 host setup · 02 environments · 03 backends · 04 metrics · 05 runbook
scripts/           host prep, one venv per backend, run_all.sh
results/           one JSON per run — the evidence behind RESULTS.md
```

## Read the docs in this order

1. [`docs/01-host-setup.md`](docs/01-host-setup.md) — power mode, swap, engine cache,
   and why a run at 15 W is not comparable to one at MAXN.
2. [`docs/02-environments.md`](docs/02-environments.md) — three venvs, and the trap
   that turns the PyTorch "GPU baseline" into a silent CPU run.
3. [`docs/03-backends.md`](docs/03-backends.md) — what each backend actually does,
   including why the monolithic ONNX cannot be built on this board and what Tether's
   ~25 ms claim would have to mean.
4. [`docs/04-metrics.md`](docs/04-metrics.md) — what every number means and what it
   does not.
5. [`docs/05-runbook.md`](docs/05-runbook.md) — the bench day, ordered so the cheapest
   failures happen first.

## Results

[`docs/RESULTS.md`](docs/RESULTS.md) — generated, with the prior claims it is meant to
replace listed until it is.

## Provenance

The fine-tuned checkpoint and the split ONNX export come from a DGX Spark (GB10); this
board only ever runs them. A TensorRT engine is hardware- and version-specific and is
**built here, never copied** — the ONNX is the portable artefact, the engine cache is
local. `bench/vendor/smolvla_split.py` is a copy of the runtime driving the real
machine, carrying its source commit and its single documented modification.
