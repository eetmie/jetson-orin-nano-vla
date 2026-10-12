# pi0.5 from LeRobot

`./export.sh pi05-base [checkpoint] [out-dir]` exports [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base)
or a fine-tune of it into the same layout as [`pi05-libero`](../pi05/README.md): seven
refittable layer templates plus the K/V-only last language layer, each layer's FP16
weights, and fixtures. It runs in `.venv-xvla` (LeRobot 0.6.1) on a CUDA machine.

```bash
./export.sh pi05-base
./export.sh pi05-base path/to/pretrained_model bundles/my-pi05 --views 2 --task "dig sand"
```

The tokenizer, [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224),
is gated: accept its terms and log in with `hf auth login` first. Both repos are pinned
in `export/hub-lock.json`.

`--task` is repeatable; the board benchmarks only tasks the bundle has. `--views`
defaults to every camera the checkpoint declares and sets the prefix capacity; the
board may feed fewer cameras, never more.

## What differs from pi05-libero

- **State in the prompt.** LeRobot's pi0.5 quantizes the normalized state into 256 bins
  and writes it into the prompt, `Task: T, State: 12 200 -1 ...;\nAction: `, so the
  prompt changes every observation. The bundle stores each task's token head and the
  token pieces of the digits; the board assembles the prompt without a tokenizer. The
  bin -1 (states below the q01 quantile) tokenizes as the merged piece `▁-` then `1`.
  The export checks the assembly against the tokenizer for every bin.
- **Prefix capacity** = `views × 256` + the worst-case prompt (task head + 4 tokens per
  state dim + 5). `pi05_base`, 3 cameras and 32 state dims: 768 + 144 = 912 tokens.
  A fine-tune with 2 cameras and a 6-dim state needs about 552.
- **FP32 language residual stream.** `pi05_base`'s language residual reaches ~6e3, where
  an FP16 step is 4. Keeping the residual adds and norms in FP32 (matmuls and K/V stay
  FP16) halved the board's mean error against stock FP32. The last layer is K/V-only:
  nothing reads its attention output or MLP, which overflows FP16.
- **Preprocessing** is LeRobot's: `x/255`, torch bilinear resize without antialias,
  centered zero padding, `x*2-1`. Quantile normalization of state and actions comes from
  the checkpoint's pre/postprocessor (identity for `pi05_base`, which has no statistics).
- **Fixtures** are stock LeRobot FP32 runs (preprocessor, `PI05Policy`, postprocessor)
  with injected noise; the last one pushes the state past the quantiles to exercise
  bins -1 and 255. The board gates against stock FP32, not against the FP16 split
  reference: see `_validate_lerobot` in `bench/vendor/pi05_trt.py` for why and the bounds.
