# Export your own checkpoint

Turns a LeRobot SmolVLA or X-VLA checkpoint (base or fine-tuned) into the split ONNX
bundle the benchmark runs. **Run it on the machine you fine-tune on**, then copy the
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
model code outside LeRobot.
