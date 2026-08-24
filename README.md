# jetson-orin-nano-vla

Does a VLA policy fit on an **8 GB Jetson Orin Nano Super**, and what does it cost?

Two model families, four runtimes, one board, the same seeded observations fed to all of
them — so the numbers can be compared rather than collected.

| model | family | params | PyTorch | split ONNX |
|---|---|---|---|---|
| `smolvla-base` | smolvla | 450 M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | [`ainekko/smolvla_base_onnx`](https://huggingface.co/ainekko/smolvla_base_onnx) |
| `xvla-base` | xvla | 880 M | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | none published — export it |

| backend | what it is |
|---|---|
| **`torch`** | stock PyTorch — the LeRobot policy straight off the checkpoint |
| **`ort-split`** | the split ONNX export on ONNX Runtime + TensorRT EP (FP16) |
| **`ort-mono`** | the *monolithic* ONNX on ORT — the split-vs-monolith A/B |
| **`tether`** | optional: [FastCrest Tether](https://github.com/FastCrest/tether) `serve` + `/act` |

Both checkpoints are Apache 2.0, so the comparison is reproducible by anyone with the
same board. A locally fine-tuned checkpoint drops in by pointing `--checkpoint` and
`--bundle` at it.

## Why

"It runs" is not the question. On this board the question is what a runtime *leaves
behind*, so every run records parity against a reference, latency, per-process CPU, RAM
resident and whole-board, and power integrated to mJ per inference. Speed with the wrong
actions is worse than slow, so parity is a first-class metric and not a footnote.

**These are best-case numbers.** The benchmark feeds observations from memory: no
cameras, no control stack, no logging, nothing else competing for the 8 GB or the six
cores. A real deployment needs headroom on top of every figure here — how much is its
own measurement, on its own rig.

## Quickstart

```bash
scripts/00_host_prep.sh                 # MAXN_SUPER, pinned clocks, persistent engine cache
scripts/10_env_torch.sh                 # .venv-torch       (asserts torch sees the GPU)
scripts/13_env_torch_xvla.sh            # .venv-torch-xvla  (lerobot 0.6.1, for X-VLA)
scripts/11_env_ort.sh                   # .venv-ort         (asserts the TensorRT EP registers)
scripts/12_env_tether.sh                # .venv-tether      (optional)

.venv-torch/bin/python -m bench selftest      # do the instruments read real numbers?

scripts/fetch_models.sh smolvla-base          # weights + the split ONNX from the Hub
MODEL=smolvla-base scripts/run_all.sh         # every backend, then docs/RESULTS.md
```

One run at a time:

```bash
python -m bench models                        # what can be benchmarked

python -m bench torch     --model smolvla-base --checkpoint ~/bundles/smolvla-base-torch \
                          --weights float32 --autocast float16 --iters 100
python -m bench ort-split --model smolvla-base --bundle ~/bundles/smolvla-base-split --views 2
python -m bench ort-mono  --model smolvla-base --onnx exports/smolvla_base_static.onnx --no-trt

python -m bench parity results
python -m bench report results --out docs/RESULTS.md
```

A locally fine-tuned policy instead of a registry entry:

```bash
python -m bench ort-split --family smolvla --bundle ~/bundles/my-split-export \
    --state-dim 3 --action-dim 4 --label mine.ort-split
```

## Docs

| | |
|---|---|
| [`01-host-setup.md`](docs/01-host-setup.md) | power mode, swap, engine cache |
| [`02-environments.md`](docs/02-environments.md) | four venvs, and the traps that silently produce a wrong number |
| [`03-backends.md`](docs/03-backends.md) | the models, the runtimes, cameras, why the monolith will not build here |
| [`04-metrics.md`](docs/04-metrics.md) | what every number means and what it does not |
| [`05-runbook.md`](docs/05-runbook.md) | the bench day, cheapest failures first |
| [`06-optimization-backlog.md`](docs/06-optimization-backlog.md) | what to try next on the split path |
| [`RESULTS.md`](docs/RESULTS.md) | generated from `results/*.json` |

## Layout

```
bench/
  models.py        the registry: families, HF artefacts, and the shapes a benchmark needs
  runner.py        warmup -> idle baseline -> measured window -> one JSON per run
  monitor.py       tegrastats parser (RAM/CPU/GPU/temp/power); psutil fallback off-board
  procwatch.py     per-PID CPU + RSS from /proc, follows children
  obs.py           deterministic observations: same images, state, task AND noise draw
  parity.py        cross-backend action-chunk comparison
  report.py        result JSONs -> the markdown tables in docs/RESULTS.md
  backends/        torch_smolvla · torch_xvla · ort_split · ort_split_xvla · ort_mono · tether_http
  vendor/          the two split runtimes, copied so the measured code is pinned
docs/  scripts/  results/
```

## Scope

Benchmarking and feasibility only. No robot, no camera, no control loop — observations
come from memory so a latency number is a property of the runtime rather than of camera
timing. Capture cost, USB bandwidth and video encode are real and are deliberately
somebody else's measurement.

TensorRT engines are hardware- and version-specific and are built on the board, never
copied; the ONNX is the portable artefact. `bench/vendor/` holds the two split runtimes
verbatim from the projects that deploy them, each carrying its source commit.

## License

This repository is **MIT** (see `LICENSE`). The models are **Apache 2.0** and stay under
their own licence — which is why `hf/xvla-base-onnx-README.md` declares `apache-2.0`.
