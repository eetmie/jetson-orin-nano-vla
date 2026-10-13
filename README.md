# jetson-orin-nano-vla

Seven vision-language-action models on an **8 GB Jetson Orin Nano Super**, running on
TensorRT with ahead-of-time compiled Triton kernels. Export a checkpoint on your PC,
copy the bundle to the Jetson, run one command.

| model | checkpoint | cameras | p50 | rate | RAM in use |
|---|---|---:|---:|---:|---:|
| SmolVLA 450M | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | 2 | 104.60 ms | 9.55 Hz | 1.65 GB |
| EVO1 775M LIBERO | [`zuoxingdong/evo1_libero`](https://huggingface.co/zuoxingdong/evo1_libero) | 2 | 333.37 ms | 3.00 Hz | 2.62 GB |
| X-VLA 0.9B | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | 3 | 342.31 ms | 2.92 Hz | 2.93 GB |
| GR00T N1.7 3B | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) | 3 (×2 frames) | 272.39 ms | 3.67 Hz | 5.72 GB |
| GR00T N1.6 3B | [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) | 3 | 272.49 ms | 3.67 Hz | 5.30 GB |
| π0.5 LIBERO 3.3B | [openpi](https://github.com/Physical-Intelligence/openpi) `pi05_libero` | 2 | 436.30 ms | 2.29 Hz | 6.30 GB |
| π0.5 base 3.3B | [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) | 3 | 642.05 ms | 1.56 Hz | 6.13 GB |

JetPack 7.2.1, MAXN_SUPER with `jetson_clocks`, 300-second runs on synthetic
observations, default denoising steps. RAM is whole-board usage during inference.
These measure inference cost, not task success. Every run first checks its actions
against the stock PyTorch policy's output stored in the bundle (cosine ≥ 0.99999,
within 0.27 % of the action range; π0.5 base 0.63 %) and refuses to benchmark otherwise.

## Use

**On the PC** (Linux, [uv](https://docs.astral.sh/uv/), enough RAM for the FP32 policy):

```bash
git clone https://github.com/eetmie/jetson-orin-nano-vla && cd jetson-orin-nano-vla
./export.sh smolvla                                   # base checkpoint -> bundles/smolvla
./export.sh smolvla path/to/pretrained_model bundles/my-smolvla --task "pick up the cube"
scp -r bundles/smolvla jetson:
```

Models: `smolvla`, `xvla`, `evo1`, `groot-n16`, `groot-n17`, `pi05-base`, `pi05-libero`.
The first export of each family builds its pinned environment. `pi05-libero` runs in
openpi's Docker container and needs the NVIDIA container toolkit. GR00T N1.7 and π0.5
base read gated tokenizers (`nvidia/Cosmos-Reason2-2B`, `google/paligemma-3b-pt-224`):
accept their terms and `hf auth login` first. More options: [export/README.md](export/README.md).

**On the Jetson** (Orin Nano Super, JetPack 7.2.1, uv):

```bash
git clone https://github.com/eetmie/jetson-orin-nano-vla && cd jetson-orin-nano-vla
./run.sh ~/smolvla
```

The first run builds the engines for that bundle (5-15 minutes) and caches them in
`~/.cache/jetson-orin-nano-vla`; later runs start in seconds.

For numbers comparable to the table, `./tune-board.sh` sets MAXN_SUPER and pins the
clocks, then offers to keep them pinned at every boot and to boot without the desktop.
It changes nothing else.

| option | |
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

How the speed was found, and what did not work: [docs/optimization.md](docs/optimization.md).

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
