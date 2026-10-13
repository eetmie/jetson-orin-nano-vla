# jetson-orin-nano-vla

A collection of vision-language-action (VLA) models on an **8 GB Jetson Orin Nano Super**, running on
TensorRT with ahead-of-time compiled Triton kernels. Export a checkpoint on your PC as a split ONNX bundle,
copy the bundle to the Jetson and run the model.

This repo is for benchmarking only. It measures how fast the models run on the board, and it does not provide the code to drive a robot with them. Use this as a base for your own projects.

| model | checkpoint | cameras | p50 | rate | RAM in use |
|---|---|---:|---:|---:|---:|
| [SmolVLA 450M](results/smolvla-base.json) | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | 2 | 107.55 ms | 9.30 Hz | 1.78 GB |
| [EVO1 775M LIBERO](results/evo1-libero.json) | [`zuoxingdong/evo1_libero`](https://huggingface.co/zuoxingdong/evo1_libero) | 2 | 332.77 ms | 3.00 Hz | 2.76 GB |
| [X-VLA 0.9B](results/xvla-base.json) | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | 3 | 342.59 ms | 2.92 Hz | 3.04 GB |
| [GR00T N1.7 3B](results/groot-n17-base.json) | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) | 3 (×2 frames) | 273.70 ms | 3.66 Hz | 5.78 GB |
| [GR00T N1.6 3B](results/groot-n16-base.json) | [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) | 3 | 273.19 ms | 3.66 Hz | 5.33 GB |
| [π0.5 LIBERO 3.3B](results/pi05-libero.json) | [openpi](https://github.com/Physical-Intelligence/openpi) `pi05_libero` | 2 | 438.47 ms | 2.28 Hz | 6.39 GB |
| [π0.5 base 3.3B](results/pi05-base.json) | [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) | 3 | 643.39 ms | 1.55 Hz | 6.18 GB |

JetPack 7.2.1, MAXN_SUPER with pinned clocks, desktop running (though headless recommended!). 300-second runs on
synthetic observations with default denoising steps from bundles exported and run with
the two commands below. RAM is whole-board usage during inference. Every run first checks its actions against the stock
policy's output stored in the bundle and refuses to benchmark otherwise: cosine ≥ 0.99999
and within 0.13 % of the action range (π0.5 LIBERO against its FP16 PyTorch conversion;
π0.5 base 0.63 % against stock FP32).

## Use

Both machines need [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`

Most of the models are too big to build as a single "monolithic" TensorRT engine on 8 GB, so here the export splits each policy into several ONNX graphs (vision, language, denoise, ...). The Jetson builds them into engines one piece at a time. At run time `bench/backends/` and `bench/vendor/trt_device.py` chain the engines together on the GPU and replay the chain as one CUDA graph. The export itself runs on a PC because it needs the whole FP32 policy in memory, though I have not tested to export directly on the Jetson. Might work with small mods, but will prob take a long time haha.

**1. Export** on your PC (Linux, enough RAM for the FP32 policy):

```bash
git clone https://github.com/eetmie/jetson-orin-nano-vla && cd jetson-orin-nano-vla
./export.sh smolvla                              # base checkpoint -> bundles/smolvla
./export.sh smolvla path/to/pretrained_model     # or your own fine-tune
```

Models: `smolvla`, `xvla`, `evo1`, `groot-n16`, `groot-n17`, `pi05-base`, `pi05-libero`.
The first export of each model sets up its environment and downloads the checkpoint.
`pi05-libero` needs Docker with the NVIDIA container toolkit. GR00T N1.7 and π0.5 base use
gated tokenizers: accept their terms on Hugging Face and `hf auth login` first.
More options: [export/README.md](export/README.md).

**2. Copy** the bundle folder to the Jetson.

**3. Run** on the Jetson:

```bash
git clone https://github.com/eetmie/jetson-orin-nano-vla && cd jetson-orin-nano-vla
./run.sh ~/smolvla
```

The first run builds the engines for that bundle (5-15 minutes) and caches them; later
runs start in seconds. For numbers comparable to the table, run `./tune-board.sh` first:
it sets MAXN_SUPER and pins the clocks, then offers to keep them pinned at every boot
and to boot without the desktop. It changes nothing else.

| `run.sh` option | |
|---|---|
| `--duration-s N` | benchmark length, default 300 |
| `--baseline` | plain TensorRT engines, without the Triton kernels |
| `--rebuild` | discard this bundle's cached engines |

Each run writes one JSON with the latency distribution and per-stage timings, RAM,
CPU/GPU load, power per rail, temperatures, the clocks the board actually ran at,
throttle events, parity against the stock policy, and package versions.

## Layout

```
export.sh, export/   PC: checkpoint -> ONNX bundle with the stock policy's reference actions
run.sh, bench/       Jetson: TensorRT runtimes per model, bench/aot/ Triton kernels, the benchmark
results/             the runs behind the table
```

How the speed was found, and what did not work: [docs/optimization.md](docs/optimization.md). There is still plenty to tune, good enough for my use!

## Thanks

To the teams behind [LeRobot and SmolVLA](https://github.com/huggingface/lerobot),
[X-VLA](https://thu-air-dream.github.io/X-VLA/), [EVO-1](https://github.com/MINT-SJTU/Evo-1),
[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) and
[openpi](https://github.com/Physical-Intelligence/openpi) for sharing their models and code;
to [FlashRT](https://github.com/flashrt-project/FlashRT) for optimization ideas; and to
Jeremy Gracey's [triton-kernel-lab](https://github.com/JeremyGracey-AI/triton-kernel-lab)
for the Orin Nano softmax the SmolVLA kernel started from.

## License

Code: MIT. Weights and exported bundles keep their upstream licenses (GR00T N1.6: NVIDIA
One-Way Noncommercial License; N1.7: NVIDIA License with a non-commercial use limitation).
