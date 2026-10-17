# Export

`./export.sh <checkpoint directory | Hugging Face URL/id> [--out DIR] [options]` turns a base or fine-tuned
checkpoint into the bundle `./run.sh` takes on the Jetson. Run it where you fine-tune:
the export holds the whole FP32 policy in memory, which the 8 GB board cannot spare.
TensorRT engines are built on the Jetson; a bundle is portable ONNX.

```bash
./export.sh smolvla                                            # lerobot/smolvla_base
./export.sh https://huggingface.co/your-name/your-model --inspect
./export.sh your-name/your-model --out bundles/my-model
./export.sh path/to/checkpoints/020000/pretrained_model --out bundles/my-model
./export.sh xvla path/to/checkpoints/020000/pretrained_model bundles/my-xvla
./export.sh groot-n16 nvidia/GR00T-N1.6-3B bundles/groot --embodiment robocasa_panda_omron
```

| option | default | |
|---|---|---|
| `--inspect` | | print the host's resolved model specification; only small Hub metadata files are fetched |
| `--out DIR` | `bundles/<checkpoint>-<identity>`; aliases keep their old directory | must be new or empty |
| `--task "..."` | `export_config.json` tasks, or a synthetic benchmark instruction | repeatable; GR00T bakes one |
| `--views N` | checkpoint's declared cameras (the base `smolvla` alias keeps 2) | baked into the graphs; must fit the checkpoint |
| `--fps N` | `export_config.json` fps, or 30 | observation rate used for benchmark headroom |
| `--revision REV` | latest commit for a custom source; `hub-lock.json` for base aliases | branches/tags resolve once to an immutable SHA, used for every download |
| `--subfolder DIR` | repository root | selects one checkpoint in a Hub repository; tree URLs also work |
| `--embodiment E` | GR00T N1.6 `robocasa_panda_omron`, N1.7 `xdof_relative_eef_relative_joint` | the embodiment head sliced in; for N1.7 it also fixes the cameras and history frames |
| `--vlm-files D` | `nvidia/Cosmos-Reason2-2B` | GR00T N1.7's tokenizer and processor files (gated repo) |

The inspector runs on the host PC, reads configuration and safetensors headers without
loading tensor data, and distinguishes real state/action dimensions from padded model
widths. It supports full checkpoints for the implemented LeRobot families: SmolVLA,
X-VLA, EVO1 and π0.5. Adapter-only checkpoints need merging into full weights first.

Each new bundle includes `model_spec.json` under its checksum manifest. The Jetson uses
that file to generate observations and select the runtime; it never inspects the source
checkpoint or imports a training framework. Incompatible AOT shapes automatically use
native TensorRT. Older bundles retain their existing behavior.

## OpenPI checkpoints

OpenPI π0.5 accepts converted `model.safetensors` checkpoints or JAX checkpoints with
`params/`; JAX conversion runs in the pinned host container. Keep the checkpoint's
`assets/` normalization statistics alongside the weights.

```bash
./export.sh path/to/pi05-libero-finetune --openpi-config pi05_libero --out bundles/my-libero
./export.sh path/to/custom-checkpoint --export-config export/openpi-example.json \
  --openpi-config-file path/to/my-training-config.py --out bundles/my-openpi
```

For one-argument usage, save the JSON observation recipe as `export_config.json` in the
checkpoint directory or Hub repository. [openpi-example.json](openpi-example.json)
shows the fields. State/action dimensions are read from normalization assets when
omitted; padded width and horizon can come from the converted model's `config.json`.
Camera names/order, raw observation keys and denoising steps must be declared for a
custom recipe. `pi05_libero` has a built-in observation recipe.

`--openpi-config NAME` selects a registered OpenPI TrainConfig and overrides the recipe's
name. For a custom TrainConfig, explicitly pass a local Python module with
`--openpi-config-file`; it must provide `register()`. The JSON metadata never implicitly
loads Python code from a model repository. `--inspect` does not import that module.

The exporter checks the declared shapes and transforms against stock OpenPI inference
with injected noise. Quantile and mean/std normalization, static prompts and discrete
state prompts are supported. Robot output transforms currently support normalization
followed by slicing to the real action width (`output_transform: "slice"`); other
transforms produce an explicit error. π0 and π0-FAST require additional exporters.

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
The shared π0.5 exporter records each framework's own stock-policy fixtures, prompt
contract and normalization. Custom exports use stock FP32 references; the locked
public OpenPI recipe retains its original mixed-precision reference.
`python -m vla_common.boundary_report <bundle>` (from `export/`) lists engine
boundaries that would be exact in FP16.

Downloads go through `vla_common.hub_download`: pinned snapshots, resumable ranges,
size and checksum verification. Each vendored exporter names its source in its first lines.
