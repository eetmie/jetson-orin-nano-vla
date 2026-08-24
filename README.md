# jetson-orin-nano-vla

Benchmarking Vision-Language-Action inference on an **8 GB NVIDIA Jetson Orin Nano
Super**, JetPack 7.2.

Two model families, four runtimes, one board, the same seeded observations fed to all
of them.

| model | family | params | PyTorch | split ONNX |
|---|---|---|---|---|
| `smolvla-base` | smolvla | 450 M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | [`ainekko/smolvla_base_onnx`](https://huggingface.co/ainekko/smolvla_base_onnx) |
| `xvla-base` | xvla | 880 M | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | none published — export it |

| backend | what it is |
|---|---|
| **`torch`** | stock PyTorch — the LeRobot policy straight off the checkpoint |
| **`ort-split`** | the split ONNX export on ONNX Runtime + TensorRT EP (FP16) |
| **`ort-mono`** | the *monolithic* ONNX on ORT — the split-vs-monolith A/B |
| **`tether`** | optional: [FastCrest Tether](https://github.com/FastCrest/tether) `serve` + `/act`, an off-the-shelf deployment CLI |

Everything is public and Apache 2.0, so the comparison is reproducible by anyone with
the same board. A locally fine-tuned checkpoint drops in by pointing `--checkpoint` and
`--bundle` at it.

## What is measured, and why those things

The interesting question on this board is not "which is fastest". It is **what a
runtime leaves behind for the rest of the robot.** So every run reports:

- **Parity** — cosine and max-abs against a reference, on identical seeded noise,
  normalized against the reference's own action range. The Orin is compute 8.7 and FP16
  is the only fast precision it has; blanket FP16 on SmolVLA is exactly what collapsed
  the SigLIP vision tower to cosine 0.805 on Blackwell. A fast, wrong backend is worse
  than a slow one.
- **Latency** — p50/p95/max, plus per-quartile drift over a sustained run, because an
  Orin at MAXN throttles and a ten-second mean hides it.
- **CPU** — per-process cores consumed, idle vs. inferring. *This is the metric that
  decides the project.* The SmolVLA split path runs six of its nine graphs on the CPU
  EP and the denoise loop in numpy; "it runs on the GPU" is a claim about three graphs,
  not a pipeline. Cores spent here are cores the control stack cannot have.
- **RAM** — resident, peak, and whole-board. The 8 GB is unified, so the system figure
  is what decides whether something gets OOM-killed. X-VLA is 2× SmolVLA's parameters
  and **2.6×** its resident memory, which is the difference between 4.9 GB of headroom
  and 1.5 GB.
- **Power** — VDD_IN at MAXN_SUPER, integrated to **mJ per inference**, the only fair
  way to compare a fast-and-hungry backend against a slow-and-frugal one.

And the figure that actually bites in a chunked policy:

```
steps_consumed_per_inference = latency_s * fps
chunk_headroom_x             = chunk_size / steps_consumed
```

These policies emit a chunk played open-loop while the next inference runs. Below 1×
the plan runs dry before its replacement lands. That is the threshold worth optimising
against — not a Hz number. (The widely quoted "SmolVLA does 10 Hz on an Orin Nano" is
an *observation* rate, not an inference rate.)

## Quickstart

```bash
scripts/00_host_prep.sh                 # MAXN_SUPER, pinned clocks, persistent engine cache
scripts/10_env_torch.sh                 # .venv-torch       (asserts torch sees the GPU)
scripts/13_env_torch_xvla.sh            # .venv-torch-xvla  (lerobot 0.6.1, for X-VLA)
scripts/11_env_ort.sh                   # .venv-ort         (asserts the TensorRT EP registers)
scripts/12_env_tether.sh                # .venv-tether

.venv-torch/bin/python -m bench selftest      # do the instruments read real numbers?

scripts/fetch_models.sh smolvla-base          # weights + the split ONNX from the Hub
MODEL=smolvla-base scripts/run_all.sh         # every backend, then docs/RESULTS.md
```

One run at a time:

```bash
python -m bench models                        # what can be benchmarked

python -m bench torch     --model smolvla-base --checkpoint ~/bundles/smolvla-base-torch \
                          --weights float32 --autocast float16 --iters 100
python -m bench ort-split --model smolvla-base --bundle ~/bundles/smolvla-base-split
python -m bench ort-mono  --model smolvla-base --onnx exports/smolvla_base_static.onnx --no-trt
python -m bench tether    --model smolvla-base --export-dir <tether export>   # optional

python -m bench parity results
python -m bench report results --out docs/RESULTS.md
```

A locally fine-tuned policy instead of a registry entry:

```bash
python -m bench ort-split --family smolvla --bundle ~/bundles/my-split-export \
    --state-dim 3 --action-dim 4 --label mine.ort-split
```

## Layout

```
bench/
  models.py        the registry: families, HF artefacts, and the shapes a benchmark needs
  runner.py        warmup -> idle baseline -> measured window -> one JSON per run
  monitor.py       tegrastats parser (RAM/CPU/GPU/temp/power); psutil fallback off-board
  procwatch.py     per-PID CPU + RSS from /proc, follows children (tether serves apart)
  obs.py           deterministic observations: same images, state, task AND noise draw
  parity.py        cross-backend action-chunk comparison
  report.py        result JSONs -> the markdown tables in docs/RESULTS.md
  backends/        torch_smolvla · torch_xvla · ort_split · ort_split_xvla · ort_mono · tether_http
  vendor/          the two split runtimes, copied so the measured code is pinned
  tools/           extract_frames.py — real frames out of a LeRobot dataset
docs/              01 host setup · 02 environments · 03 backends · 04 metrics ·
                   05 runbook · 06 optimization backlog
scripts/           host prep, one venv per backend, fetch_models.sh, run_all.sh
results/           one JSON per run — the evidence behind RESULTS.md
```

## Read the docs in this order

1. [`docs/01-host-setup.md`](docs/01-host-setup.md) — power mode, swap, engine cache,
   and why a run at 15 W is not comparable to one at MAXN.
2. [`docs/02-environments.md`](docs/02-environments.md) — four venvs, and the two traps
   that turn a GPU baseline into a silent CPU run or an unimportable lerobot.
3. [`docs/03-backends.md`](docs/03-backends.md) — the models, the runtimes, and why
   the monolithic ONNX cannot be built on this board.
4. [`docs/04-metrics.md`](docs/04-metrics.md) — what every number means and what it
   does not.
5. [`docs/05-runbook.md`](docs/05-runbook.md) — the bench day, ordered so the cheapest
   failures happen first.
6. [`docs/06-optimization-backlog.md`](docs/06-optimization-backlog.md) — what to try
   next on the split path, in the order that keeps each result interpretable.

## Results

[`docs/RESULTS.md`](docs/RESULTS.md) — generated, with the prior claims it is meant to
replace listed until it is.

## Provenance

TensorRT engines are hardware- and version-specific and are **built on the board, never
copied**; the ONNX is the portable artefact. `bench/vendor/` holds the two split
runtimes verbatim from the projects that deploy them, each carrying its source commit
and its single documented modification, so the code that produced a number is pinned
next to that number.
