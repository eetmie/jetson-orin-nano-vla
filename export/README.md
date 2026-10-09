# Export your own checkpoint

Turns a LeRobot SmolVLA or X-VLA checkpoint (base or fine-tuned), or NVIDIA's GR00T
N1.6, into the split ONNX bundle the benchmark runs. **Run it on the machine you fine-tune on**, then copy the
bundle to the Jetson:

```
fetch model -> (fine-tune) -> export/export.sh -> copy bundle -> benchmark on the Orin
```

The export needs the whole FP32 policy in memory plus a trace of each graph. That fits
a training machine easily and the 8 GB board only barely, so the board's job starts at
the bundle. TensorRT engines are still built on the Jetson: they are specific to the
device and are never copied.

## Setup

Linux with Python 3.12. A GPU is not needed.

```bash
export/setup.sh            # .venv-smolvla (lerobot 0.5.1) and .venv-xvla (lerobot 0.6.1)
```

## Export

```bash
export/export.sh lerobot/smolvla_base ~/bundles/smolvla-base-split --views 2
export/export.sh path/to/checkpoints/020000/pretrained_model ~/bundles/my-xvla
```

| option | default | |
|---|---|---|
| `--views N` | the checkpoint's `observation.images.*` count | baked into the graphs; the runtime may feed fewer cameras, never more |
| `--task "..."` | the dataset's tasks, if the checkpoint has them | repeatable; written into the bundle |
| `--fps N` | the dataset's fps | written into the bundle |

SmolVLA bundles stay FP32 and TensorRT builds FP16 engines from them. X-VLA bundles get
a mixed-FP16 weight pass (LayerNorm and Softmax kept FP32), which halves what stays
resident on the board. Every bundle carries a `MANIFEST.sha256`; check it after copying:
`cd <bundle> && sha256sum -c MANIFEST.sha256`.

## GR00T N1.6

Not a LeRobot policy, so it has its own venv and NVIDIA's model code (Isaac-GR00T
n1.6.1-release, fetched at a pinned commit):

```bash
export/setup.sh groot
export/export.sh nvidia/GR00T-N1.6-3B ~/bundles/groot-n16-base-split
```

| option | default | |
|---|---|---|
| `--embodiment E` | `robocasa_panda_omron` | which of the checkpoint's embodiment heads is sliced in |
| `--views N` | 3 | cameras, baked into the graphs |
| `--task "..."` | `pick up the red cube and place it in the bowl` | one prompt, baked as token ids |

The stock PyTorch model runs first on seeded inputs and its outputs ship in the bundle
as `fixture.npz`; the board checks its engines against them before measuring. On the
Spark this took under 3 minutes with the checkpoint cached. The weights are under the
NVIDIA One-Way Noncommercial License, and so is the bundle.

## Benchmark it

On the Jetson, use the base profile of the same family with your bundle:

```bash
.venv-ort/bin/python -m bench ort-split --model smolvla-base \
    --bundle ~/bundles/my-smolvla --views 1 --task "pick up the cube" --label my-smolvla.ort
```

To check the conversion against the PyTorch policy, copy the checkpoint too, run
`bench torch --checkpoint <dir>` with the same `--views` and `--task`, and compare the two
runs with `python -m bench parity` (see the README's parity gate).

## Provenance

The exporters are vendored from the author's fine-tuning pipeline. Each file names its
source path and commit in its first lines. EVO1 is not included: its exporter depends on
model code outside LeRobot. GR00T's does too, but that code is NVIDIA's public repository,
which `setup.sh groot` fetches.
