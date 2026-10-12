# Export

`./export.sh <model> [checkpoint] [out-dir] [options]` turns a base or fine-tuned
checkpoint into the bundle `./run.sh` takes on the Jetson. Run it where you fine-tune:
the export holds the whole FP32 policy in memory, which the 8 GB board cannot spare.
TensorRT engines are built on the Jetson; a bundle is portable ONNX.

```bash
./export.sh smolvla                                            # lerobot/smolvla_base
./export.sh xvla path/to/checkpoints/020000/pretrained_model bundles/my-xvla
./export.sh groot-n16 nvidia/GR00T-N1.6-3B bundles/groot --embodiment robocasa_panda_omron
```

| option | default | |
|---|---|---|
| `--task "..."` | the checkpoint's dataset tasks | repeatable; GR00T bakes one |
| `--views N` | the checkpoint's `observation.images.*` count (SmolVLA: 2) | baked into the graphs; the runtime may feed fewer cameras, never more |
| `--fps N` | the dataset's fps | |
| `--revision SHA` | `hub-lock.json` for the base checkpoints | full commit SHA; required for other Hub checkpoints |
| `--embodiment E` | GR00T N1.6 `robocasa_panda_omron`, N1.7 `xdof_relative_eef_relative_joint` | the embodiment head sliced in; for N1.7 it also fixes the cameras and history frames |
| `--vlm-files D` | `nvidia/Cosmos-Reason2-2B` | GR00T N1.7's tokenizer and processor files (gated repo) |

The SmolVLA AOT kernels need two camera slots; a one-camera fine-tune exported with
`--views 1` runs with `./run.sh --baseline`.

## Environments

The first export of each family creates its environment with uv, pinned to the
dependency closures in `constraints-*.txt`:

| environment | models | |
|---|---|---|
| `.venv-smolvla` | SmolVLA | lerobot 0.5.1 |
| `.venv-xvla` | X-VLA, EVO1, π0.5 base | lerobot 0.6.1; π0.5 base exports on CUDA |
| `.venv-groot`, `.venv-groot17` | GR00T N1.6, N1.7 | NVIDIA's Isaac-GR00T at a pinned commit |
| Docker | π0.5 LIBERO | openpi's pinned container ([details](pi05/README.md)) |

Recreate one with `export/setup.sh <environment>`.

## What a bundle holds

Mixed-FP16 graphs (norms and softmax stay FP32), the tokenizer or baked token ids, the
stock policy's FP32 output for seeded inputs (`fixture.npz`), which the Jetson checks its
engines against before measuring, and a `MANIFEST.sha256` over every file.
`python -m vla_common.boundary_report <bundle>` (from `export/`) lists engine
boundaries that would be exact in FP16.

Downloads go through `vla_common.hub_download`: pinned snapshots, resumable ranges,
size and checksum verification. Each vendored exporter names its source in its first lines.
