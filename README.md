# jetson-orin-nano-vla

**Tested on JetPack 7.2.1 (L4T R39.2.1).**

Recipes and measurements for running VLA models on an **8 GB Jetson Orin Nano Super**:
four public base checkpoints and the trained EVO1 LIBERO and π0.5 LIBERO checkpoints,
all on a pure TensorRT runtime (`bench trt-split`).

| model | upstream checkpoint |
|---|---|
| SmolVLA 450M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| EVO1 775M LIBERO | [`zuoxingdong/evo1_libero`](https://huggingface.co/zuoxingdong/evo1_libero) |
| X-VLA 0.9B | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) |
| GR00T N1.7 3B | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| GR00T N1.6 3B | [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) |
| π0.5 LIBERO 3.3B (prototype) | [openpi](https://github.com/Physical-Intelligence/openpi) `pi05_libero` |

## Measured fit

Pinned MAXN_SUPER clocks, deterministic synthetic observations, default denoising steps.
These measure inference cost, not robot-task quality.

| model / runtime | views | p50 | p95 | rate | RAM in use |
|---|---:|---:|---:|---:|---:|
| **SmolVLA** TensorRT + AOT Triton, mixed FP16, lean runtime | 2 | 102.85 ms | 103.54 ms | 9.70 Hz | 1.63 GB |
| EVO1 LIBERO split, ORT mixed FP16 | 2 | 414.67 ms | 424.72 ms | 2.41 Hz | 6.00 GB |
| **EVO1 LIBERO** TensorRT + AOT Triton, mixed FP16 | 2 | 340.72 ms | 341.53 ms | 2.94 Hz | 2.53 GB |
| X-VLA PyTorch FP32 | 3 | 2313.50 ms | 2320.89 ms | 0.43 Hz | 5.45 GB |
| X-VLA split, ORT FP16 | 3 | 391.55 ms | 407.33 ms | 2.55 Hz | 5.39 GB |
| **X-VLA** TensorRT + AOT Triton, mixed FP16 | 3 | 349.31 ms | 350.36 ms | 2.86 Hz | 2.82 GB |
| **GR00T N1.7 3B** split, pure TensorRT mixed FP16 | 3 (×2 frames) | 279.67 ms | 280.50 ms | 3.58 Hz | 5.82 GB |
| **GR00T N1.6 3B** split, pure TensorRT mixed FP16 | 3 | 276.49 ms | 277.20 ms | 3.62 Hz | 5.38 GB |
| π0.5 LIBERO split, pure TensorRT FP16, padded prefix | 2 | 684.93 ms | 686.16 ms | 1.46 Hz | 6.34 GB |
| **π0.5 LIBERO** split, pure TensorRT FP16, compact prefix | 2 | 447.64 ms | 447.93 ms | 2.23 Hz | 6.29 GB |

The pure TensorRT rows come from one board, the PyTorch and ORT rows from a second one
with the same JetPack and clock settings, which runs X-VLA and EVO1 a few percent faster.
YMMV. `RAM in use` is the whole system while inferring.
The SmolVLA row is the selected configuration from a 300-second run; its process RSS
averages 1.40 GB. [Its step-by-step comparison](results/smolvla-native-20261010T1247Z/summary.json)
and [the earlier speed and RAM comparisons](results/smolvla-memory-20261010/summary.json)
are retained. The X-VLA and EVO1 rows use the [X-VLA](experiments/xvla_triton/) and
[EVO1](experiments/evo1_triton/) experiments; every bold row from EVO1 down is a
300-second run from [the all-model round](results/all-models-20261010/), with exact
host preprocessing and FP16 engine boundaries. The π0.5 padded-prefix row predates it.
[The playbook](docs/07-optimization-playbook.md) lists what was found and what each
model still has to go through.

## What every run logs

Each run writes one JSON to [`results/`](results/), and `python -m bench report` turns
them into [the results page](docs/RESULTS.md). Besides the latency distribution, a run
records:

- per-stage timings, and the first call after load;
- whole-board RAM and swap, process RSS, CPU and GPU load;
- power per rail and energy per inference;
- temperatures, and drift over a 5-minute sustained run;
- the clocks the board actually ran at (every CPU core, GPU, memory) next to the ones it
  was set to, plus over-current and thermal throttle events;
- action parity against the stock PyTorch policy;
- the board state: JetPack, power mode, package versions, repo commit.

[The profile](docs/06-profile.md) adds an Nsight Systems trace of every model: where each
stage's time goes and whether it is compute- or bandwidth-bound.

The selected [SmolVLA configuration](experiments/smolvla_triton/#layout-cross-attention-kv-and-rope)
combines fused vision attention, packed expert gated projections, cross-attention K/V
computed once per observation (`export.sh --hoist-cross-kv`), image scaling on the GPU,
a lighter tokenizer and a bounded prompt cache. It uses an explicitly verified
experimental engine cache; follow that reproduction command to obtain the row above.
The ordinary ONNX build path retains its standard TensorRT engines. Weights and
activations stay FP16 with norms and softmax in FP32; the custom kernels accumulate in
FP32, while TensorRT chooses FP16-accumulating kernels for many of its own GEMMs.

## Parity

Every pure TensorRT model reproduces the stock PyTorch FP32 actions to **cosine 0.99999
or better, within 0.27 % of the action range over the whole action chunk**. π0.5 is
checked the same way against its FP16 PyTorch conversion, which itself sits at cosine
0.99998 from the original mixed-precision policy.

Each bundle carries the stock policy's output for seeded inputs and noise, and loading
refuses to run if the engines miss it. Per-model values are in
[the results](docs/RESULTS.md#parity); `python -m bench parity` compares any two runs.

## Run it

Export on the machine you fine-tune on, copy the bundle to the Orin whole, and run it
there. Engines are built on the Orin on the first run and cached.

```bash
# on the export machine
export/setup.sh                       # SmolVLA, X-VLA and EVO1
export/export.sh lerobot/smolvla_base ~/bundles/smolvla-base-split --views 2
export/export.sh lerobot/xvla-base ~/bundles/xvla-base-split
export/export.sh zuoxingdong/evo1_libero ~/bundles/evo1-libero-split \
    --task "pick up the black bowl and place it on the plate"

export/setup.sh groot                 # GR00T N1.6
export/export.sh nvidia/GR00T-N1.6-3B ~/bundles/groot-n16-base-split
export/setup.sh groot17               # GR00T N1.7
export/export.sh nvidia/GR00T-N1.7-3B ~/bundles/groot-n17-base-split

# on the Orin
scripts/00_host_prep.sh
scripts/11_env_ort.sh
MODEL=smolvla-base   BUNDLE=~/bundles/smolvla-base-split   scripts/run_all.sh
MODEL=groot-n16-base BUNDLE=~/bundles/groot-n16-base-split scripts/run_all.sh
```

`run_all.sh` also runs the PyTorch reference for SmolVLA and X-VLA when its venv exists
(`scripts/10_env_torch.sh`, `scripts/13_env_torch_xvla.sh`) and the checkpoint has been
fetched (`python -m bench fetch --model <model> --what torch`).

- GR00T N1.7 reads its tokenizer from the gated
  [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B): accept
  its terms on Hugging Face before exporting. It sees every camera now and 30 frames
  earlier; the runtime reuses the earlier frame's encode.
- `evo1-libero` and `pi05-libero` actions mean something for LIBERO's embodiment only.
- π0.5 is a prototype: its export (`export/pi05/`) runs in the openpi container and is
  not an `export.sh` entry yet. The compact prefix packs only the real camera views and
  prompt tokens, and fits short prompts.
- The ORT rows ran the Hugging Face bundles ([`eetmie/smolvla-base-onnx`](https://huggingface.co/eetmie/smolvla-base-onnx),
  [`eetmie/xvla-base-onnx`](https://huggingface.co/eetmie/xvla-base-onnx)) with
  `bench ort-split`. Building X-VLA's ORT engines needs 4 GB swap and a freshly booted,
  headless board ([host setup](docs/01-host-setup.md#swap--4-gb-and-build-on-a-freshly-booted-headless-board)).

`export/` also takes your own fine-tuned LeRobot checkpoint; see
[export/README.md](export/README.md).

## Documentation

- [Host setup](docs/01-host-setup.md)
- [Python environments](docs/02-environments.md)
- [Model and runtime contracts](docs/03-backends.md)
- [Metric definitions](docs/04-metrics.md)
- [Benchmark runbook](docs/05-runbook.md)
- [Profile: where the time goes](docs/06-profile.md)
- [Measured results](docs/RESULTS.md)

## Thanks

Big thanks to the teams behind [LeRobot and SmolVLA](https://github.com/huggingface/lerobot),
[X-VLA](https://thu-air-dream.github.io/X-VLA/), [EVO-1](https://github.com/MINT-SJTU/Evo-1),
[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T), and
[Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi) for
sharing their models and code. Thanks also to [FlashRT](https://github.com/flashrt-project/FlashRT)
for deployment and optimization ideas, and to Jeremy Gracey's
[triton-kernel-lab](https://github.com/JeremyGracey-AI/triton-kernel-lab) for the
Orin Nano softmax examples used in the [initial Triton experiments](experiments/smolvla_triton/).

## Scope

Inference cost and parity on the board. No training, fine-tuning, robot control or
camera capture. TensorRT engines are built on the Jetson and never copied between
machines; the ONNX bundles are the portable artifacts.

Repository code is MIT. Model weights and derived exports keep their upstream licenses
(GR00T N1.6: NVIDIA One-Way Noncommercial License; N1.7: NVIDIA License with a
non-commercial use limitation); consult each model card before redistribution.
