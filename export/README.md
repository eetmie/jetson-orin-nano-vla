# Export your own checkpoint

Turns a LeRobot SmolVLA, X-VLA or EVO1 checkpoint (base or fine-tuned), or NVIDIA's
GR00T N1.6/N1.7, into the split ONNX bundle the benchmark runs. **Run it on the machine you fine-tune on**, then copy the
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

EVO1 uses `.venv-xvla` (it is in LeRobot 0.6.1 too).

## Export

```bash
export/export.sh lerobot/smolvla_base ~/bundles/smolvla-base-split --views 2
export/export.sh path/to/checkpoints/020000/pretrained_model ~/bundles/my-xvla
export/export.sh zuoxingdong/evo1_libero ~/bundles/evo1-libero-split
```

| option | default | |
|---|---|---|
| `--views N` | the checkpoint's `observation.images.*` count | baked into the graphs; the runtime may feed fewer cameras, never more |
| `--task "..."` | the dataset's tasks, if the checkpoint has them | repeatable; written into the bundle |
| `--fps N` | the dataset's fps | written into the bundle |
| `--hoist-cross-kv` | off | SmolVLA: prefill emits the cross-attention layers' expert K/V once per observation instead of decode re-projecting them every step, and the KV cache crosses engines in FP16. `trt-split` only |

Every bundle gets mixed-FP16 graphs, which halve what stays resident on the board:
X-VLA keeps LayerNorm and Softmax FP32; SmolVLA's three large graphs and every EVO1
engine graph also keep each RMSNorm FP32. Each bundle also carries the stock policy's
FP32 output for seeded inputs (`fixture.npz`; EVO1 `parity_fixture.npz`), which
`bench trt-split` checks its engines against at load, and SmolVLA and EVO1 bundles carry
their token embedding as `embed_tokens.npy`. EVO1 fetches its VLM base,
`OpenGVLab/InternVL3-1B-hf`, at the pinned revision its LIBERO recipe trained from.
SmolVLA's RoPE and patch embedding are exported as concatenation and patchify + MatMul
(the same arithmetic as LeRobot's slice-assign RoPE and stride-16 conv, which TensorRT
runs as unfused scatters and a slow convolution kernel).
Every bundle carries a `MANIFEST.sha256`; check it after copying:
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

## GR00T N1.7

Own venv (transformers 4.57.3) and NVIDIA's n1.7-release model code:

```bash
export/setup.sh groot17
export/export.sh nvidia/GR00T-N1.7-3B ~/bundles/groot-n17-base-split
```

| option | default | |
|---|---|---|
| `--embodiment E` | `xdof_relative_eef_relative_joint` | which pretrained embodiment head is sliced in; it also fixes the cameras and history frames |
| `--task "..."` | `pick up the red cube and place it in the bowl` | one prompt, baked as token ids |
| `--vlm-files D` | `nvidia/Cosmos-Reason2-2B` | where the tokenizer, chat template and image-processor files come from |

`nvidia/Cosmos-Reason2-2B` is gated on Hugging Face: accept its terms, or pass
`--vlm-files` with a local copy. Only those small files are read; the backbone weights
come from the GR00T checkpoint. The export took about 3 minutes on the Spark. The
checkpoint's license file is the NVIDIA License with a non-commercial use limitation.

## Benchmark it

On the Jetson, use the base profile of the same family (EVO1: `evo1-libero`) with your
bundle:

```bash
.venv-ort/bin/python -m bench trt-split --model smolvla-base \
    --bundle ~/bundles/my-smolvla --views 1 --task "pick up the cube" --label my-smolvla.trt
```

`ort-split` runs the same bundles through ONNX Runtime, except `--hoist-cross-kv` ones.

To check the conversion against the PyTorch policy, copy the checkpoint too, run
`bench torch --checkpoint <dir>` with the same `--views` and `--task`, and compare the two
runs with `python -m bench parity` (see the README's parity gate).

## Provenance

The exporters are vendored from the author's fine-tuning pipeline. Each file names its
source path and commit in its first lines. GR00T's depends on model code outside
LeRobot: NVIDIA's public repository, which `setup.sh groot` / `setup.sh groot17` fetches.
