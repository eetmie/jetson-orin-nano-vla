# 3. What is actually being compared

Three runtimes, one model, one board. The model is SmolVLA 450M fine-tuned on the
excavator digging dataset: **chunk 12, 10 denoise steps, one IR camera, 3-joint state,
4-dim action** (joystick rate commands in [-1, 1]).

## `torch` — stock PyTorch, LeRobot `SmolVLAPolicy`

The control baseline. Load the checkpoint the way the training repo does, call
`policy.model.sample_actions(...)`. No export, no engine, no build step. Everything
the other two backends buy has to be paid for against this.

Three variants, because "FP16 PyTorch" is ambiguous and the difference matters on an
8 GB board:

| label | flags | what it is |
|---|---|---|
| `torch-fp32` | `--weights float32 --autocast off` | the numerical reference |
| `torch-amp16` | `--weights float32 --autocast float16` | mixed precision as LeRobot itself runs it (`use_amp: true`); FP16 matmuls, FP32 master weights |
| `torch-half16` | `--weights float16 --patch-half-out` | hard cast — halves resident weights |

`torch-half16` needs a patch. LeRobot 0.5.1 hardcodes
`suffix_out.to(dtype=torch.float32)` inside `denoise_step`, so a half-weights policy
dies with *"mat1 and mat2 must have the same dtype"*. `--patch-half-out` wraps that one
projection to cast to its own weight dtype. It is a deliberate deviation from stock
LeRobot and is recorded in the run metadata as `patched_half_out: true`. Measured on a
GB10 against `torch-fp32`, the half cast came out at cosine 0.999999 / max abs diff
1.2e-3 — no vision-tower collapse — but that is a Blackwell result and needs
re-confirming on Orin, which is exactly what `bench parity` is for.

`torch.compile` is available behind `--compile` and is off by default: compile time on
this board is minutes and the point of the baseline is what you get out of the box.

## `ort-split` — the incumbent, and why it looks the way it does

The split 9-graph ONNX export on ONNX Runtime with the TensorRT execution provider,
FP16, flow-matching denoise loop in numpy. This is the path already driving the
machine (`kaivuriprokkis/lerobot_vla/smolvla_split.py`, vendored under
`bench/vendor/` so the measured code is pinned next to its numbers).

It is split because the monolithic export **cannot be built on this board**. That is
measured, not assumed: the 108k-node 10-step graph OOM'd or errored out on every
combination of FP16/FP32, GUI/headless, and aggressive TRT knobs, taking up to 85
minutes to fail. Cutting to 5 steps (61k nodes) barely moved the peak — 6.7 GB against
7.4 GB — because the floor is node-count independent: TensorRT imports all 450M
weights as FP32 working copies. Splitting means each engine carries only its own
weight slice, and each then builds in under a minute.

What that costs, and why this backend reports a per-graph breakdown:

| graph | runs on | calls per inference |
|---|---|---|
| `smolvlm_vision` | TensorRT EP | 1 |
| `smolvlm_expert_prefill` | TensorRT EP | 1 |
| `smolvlm_expert_decode` | TensorRT EP | **10** (once per denoise step) |
| `smolvlm_text` | CPU EP | 1 (cached per instruction) |
| `state_projector` | CPU EP | 1 |
| `action_in`, `time_in`, `time_out`, `action_out` | CPU EP | **10 each** |

Plus the Euler update and the attention masks in numpy. On a board whose whole
selling point here is leaving CPU headroom for the robot control stack, that CPU-side
loop is the thing to measure rather than assume — so every ORT session is wrapped in
a timer and the result JSON carries `graphs_gpu`, `graphs_cpu` and `python_numpy`
separately.

First run builds three engines, one subprocess per graph (two builds in one process
OOM 8 GB). Budget ~5 minutes cold; afterwards it loads from cache in seconds.

## `tether` — the third-party claim under test

[FastCrest Tether](https://github.com/FastCrest/tether) exports, verifies and serves
VLA policies, with SmolVLA among six supported families. Its hardware table lists
**SmolVLA on an 8 GB Orin Nano at ~25 ms FP16**.

That number is the reason this repo exists. It is roughly an order of magnitude below
what the split TensorRT path projects for the same work on the same board (~110 ms at
5 steps, ~170 ms at 10, from measured per-graph timings: vision 33 ms, prefill 16.5 ms,
decode 11.4 ms per step). Both cannot describe the same computation, so one of these
is true:

- it really is that fast — adaptive step early-exit, CUDA graphs, or the Triton fast
  kernels doing work the split path does not;
- the figure is a single forward pass rather than the full 10-step denoise loop;
- the monolithic export falls back to the CUDA EP, which is the ~500 ms regime the
  monolith already measured here; or
- it does not build in 8 GB at all, which is what happened to every other monolithic
  TensorRT build attempted on this board.

Every one of those is a result. The backend records the outcome instead of treating a
failure as a crash, and `--startup-timeout` defaults generously because a first
TensorRT build is minutes at best.

Two things this backend measures around:

1. **It is a server.** The model lives in another process, so CPU/RSS attribution
   follows that PID, and the system-wide tegrastats figures are the ones that count.
2. **The timing includes transport.** `total` is the client roundtrip — PNG encode,
   JSON, loopback HTTP. Where the response carries a server-side figure it is
   reported separately as `server`. See `docs/04-metrics.md`.

Tether's `/act` request schema is not pinned in its public docs, so the first call
negotiates: several payload shapes are tried and the accepted one is recorded in the
run metadata. If none work:

```bash
python -m bench tether-probe --url http://127.0.0.1:8000   # prints its OpenAPI schema
python -m bench tether --payload-template my_payload.json ...
```

### The export question

Tether's fast path is a **monolithic** ONNX with the denoise loop unrolled — the same
shape of graph that will not TRT-build here. Exporting it is also heavy. If
`tether export` cannot run on the Orin, export on the DGX Spark and rsync the export
directory over; a TensorRT engine is not portable between the two, but the ONNX is.
Note this in the run's `--notes` so the result carries its own provenance.
