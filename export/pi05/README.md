# pi0.5 LIBERO export (prototype)

Physical Intelligence's `pi05_libero` ([openpi](https://github.com/Physical-Intelligence/openpi)),
exported as FP16 layer templates plus each layer's own weights, for `bench trt-split
--model pi05-libero`. These are the prototype's scripts as they ran. They have not been
folded into `export/export.sh` yet.

## Environment

The export runs in the openpi container built from `Dockerfile` (NGC PyTorch 25.09,
openpi's patched `transformers` 4.53.2, CPU-only JAX). Inside it, with the checkpoint
directory mounted at `/workspace/checkpoints`:

```bash
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero/* /workspace/checkpoints/pi05_libero/
python /opt/openpi/examples/convert_jax_model_to_pytorch.py --config_name pi05_libero \
    --checkpoint_dir /workspace/checkpoints/pi05_libero \
    --output_path /workspace/checkpoints/pi05_libero_pytorch
cp -r /workspace/checkpoints/pi05_libero/assets /workspace/checkpoints/pi05_libero_pytorch/
```

## Stages

Each script writes beside itself; the stages read each other by directory name.

| stage directory | scripts | writes |
|---|---|---|
| `orin_initial_20261010_001/` | `prepare_blocks.py`, `prepare_conditioning.py` | five reference fixtures from the stock mixed BF16/FP32 policy (images, prompts, injected noise, actions); the fixed ten-step AdaRMS modulation table |
| `pi05_fp16_full_20261010/` | `prepare_full.py`, `export_utils.py` | the padded bundle: 968-token prefix, three image slots |
| `pi05_fp16_compact_20261010/` | `prepare_compact.py` (with the first stage copied to `reference_initial/` and the second's bundle to `reference_baseline/`) | the compact bundle: real cameras and real prompt tokens only, 521-token prefix |

A bundle is `bundle.json`, `templates/<kind>.onnx` (one per layer kind: stem, vision,
vision_tail, language, action_input, action, action_output), `weights/<component>/` (67
components), the FP16 embedding table the runtime memory-maps, the conditioning table,
RoPE tables, `norm_stats.json` and `fixtures/` (full-FP16 PyTorch trajectories the board
checks its engines against at load). The runtime also needs a `tasks` table of prompts
tokenized for this bundle (`tokens`, `mask`, 200 each); the prototype's fixtures supply
five.

The compact bundle's prefix holds the two camera views (512 rows) plus 9 prompt tokens:
enough for the fixture prompts, not for longer ones. Longer prompts need a larger
capacity and new language/action templates.

The checkpoint's license is Physical Intelligence's; consult openpi before
redistributing a bundle.
