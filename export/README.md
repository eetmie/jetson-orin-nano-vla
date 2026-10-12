# Export your own checkpoint

Turns a LeRobot SmolVLA, X-VLA, EVO1 or pi0.5 checkpoint (base or fine-tuned), or NVIDIA's
GR00T N1.6/N1.7 or the openpi pi0.5 LIBERO policy, into the split ONNX bundle the benchmark runs. **Run it on the machine you fine-tune on**, then copy the
bundle to the Jetson ([LeRobot pi0.5 notes](pi05_lerobot/README.md)):

```
fetch model -> (fine-tune) -> export/export.sh -> copy bundle -> benchmark on the Orin
```

The export needs the whole FP32 policy in memory plus a trace of each graph. That fits
a training machine easily and the 8 GB board only barely, so the board's job starts at
the bundle. TensorRT engines are still built on the Jetson: they are specific to the
device and are never copied.

## Setup

Linux with Python 3.12. The LeRobot/GR00T exporters run on CPU, except LeRobot pi0.5
(CUDA, `.venv-xvla`); openpi pi0.5 LIBERO uses a pinned
Docker container and needs NVIDIA GPU support. Follow the
[all-model recipe](../docs/10-all-model-setup.md) for the six base checkpoints.

```bash
export/setup.sh            # .venv-smolvla (lerobot 0.5.1) and .venv-xvla (lerobot 0.6.1)
```

EVO1 uses `.venv-xvla` (it is in LeRobot 0.6.1 too).

## Export

For the selected SmolVLA Triton runtime, use
`--views 2 --hoist-cross-kv`. Known remote checkpoints and their auxiliary models
default to the full commit SHAs in `hub-lock.json`. Custom remote checkpoints need
`--revision <full-checkpoint-commit-sha>`. HF revisions use separate download
directories; local checkpoint paths do not accept `--revision`.

```bash
export/export.sh lerobot/smolvla_base ~/bundles/smolvla-base-split --views 2 --hoist-cross-kv
export/export.sh path/to/checkpoints/020000/pretrained_model ~/bundles/my-xvla
export/export.sh zuoxingdong/evo1_libero ~/bundles/evo1-libero-split
```

| option | default | |
|---|---|---|
| `--views N` | the checkpoint's `observation.images.*` count | baked into the graphs; the runtime may feed fewer cameras, never more |
| `--task "..."` | the dataset's tasks, if the checkpoint has them | repeatable; written into the bundle |
| `--fps N` | the dataset's fps | written into the bundle |
| `--hoist-cross-kv` | off | SmolVLA: prefill emits the cross-attention layers' expert K/V once per observation instead of decode re-projecting them every step, and the KV cache crosses engines in FP16. Required for the selected optimized setup. `trt-split` only |
| `--revision SHA` | `hub-lock.json` | Full 40-character HF commit SHA; moving tags are rejected. |

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
`python -m vla_common.boundary_report <bundle>` lists FP32 tensors that cross engines
and would be exact in FP16; `python -m vla_common.half_boundary --bundle <dir> --names
REGEX` switches them in an existing bundle (run from `export/`, any exporter venv).
EVO1 bundles keep their action key/value cache FP16 by default.
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
scripts/12_env_trt.sh
scripts/14_env_triton.sh
python3 -m bench.tools.vla_setup --model smolvla-base \
    --bundle ~/bundles/my-smolvla --work-dir ~/vla-runs/my-smolvla-new
```

`ort-split` runs the same bundles through ONNX Runtime, except `--hoist-cross-kv` ones.
The optimized setup supports the documented base-model shapes; custom shapes can
use the ordinary backend directly with explicit views/task and a separate cache.

To check the conversion against the PyTorch policy, copy the checkpoint too, run
`bench torch --checkpoint <dir>` with the same `--views` and `--task`, and compare the two
runs with `python -m bench parity` (see the README's parity gate).

## Provenance

Complete exporter dependency closures are fixed in `constraints-*.txt`. Nested
Hub reads use `vla_common.run_pinned`, which fixes both downloads and Transformers
cache lookups to the lock. GR00T source uses sparse Git checkouts at full commits.
The [pi0.5 exporter](pi05/README.md) pins its image digest, openpi source commit,
raw GCS object generations, tokenizer and trained tensor contents.

The exporters are vendored from the author's fine-tuning pipeline. Each file names its
source path and commit in its first lines. GR00T's depends on model code outside
LeRobot: NVIDIA's public repository, which `setup.sh groot` / `setup.sh groot17` fetches.

Remote Hub files use `vla_common.hub_download`: immutable snapshot metadata,
bounded HTTP ranges and transient retries, then size and LFS/Git-blob checksum
verification before publication. Auxiliary processor/model files are preloaded
at the pinned SHA into the Hub cache. This avoids relying on Xet transfers.
Existing completed files must still match. Interrupted ranges resume after
checking their saved chunk hashes; partial files without range metadata restart.
