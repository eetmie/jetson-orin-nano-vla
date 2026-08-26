# 5. Benchmark runbook

Run the cheapest checks first. Keep one model process active at a time because CPU and
GPU share the same 8 GB.

## 1. Fetch or copy matching artifacts

```bash
scripts/fetch_models.sh smolvla-base
scripts/fetch_models.sh xvla-base    # checkpoint only
```

For a fine-tuned policy, copy `pretrained_model/` and its split export. Confirm
`export_info.json`, tokenizer, statistics, and checkpoint identify the same training
artifact. Do not copy optimizer state or TensorRT engines.

## 2. Prepare and verify the board

```bash
scripts/00_host_prep.sh
scripts/00_host_prep.sh --verify
```

The expected state is MAXN_SUPER, pinned CPU/GPU/EMC clocks, working swap, and a
persistent engine cache. Close memory-heavy applications before a cold build.

## 3. Build the environments

```bash
scripts/10_env_torch.sh
scripts/13_env_torch_xvla.sh    # X-VLA only
scripts/11_env_ort.sh
```

Then prove the monitor is collecting real Jetson data:

```bash
.venv-torch/bin/python -m bench selftest --seconds 4
```

Expect `TegrastatsMonitor` and populated power/GPU fields.

## 4. Record the Torch reference

```bash
M=smolvla-base
CKPT=~/bundles/$M-torch
BUNDLE=~/bundles/$M-split

.venv-torch/bin/python -m bench torch --model "$M" --checkpoint "$CKPT" \
    --weights float32 --autocast off --iters 30 --label "$M.torch-fp32"
```

For X-VLA use `.venv-torch-xvla`. If the reference cannot run, record the failure and
do not claim on-device elementwise parity.

## 5. Run the split deployment path

The first invocation builds engines serially and can take several minutes. Later runs
reuse the persistent cache.

```bash
.venv-ort/bin/python -m bench ort-split --model "$M" --bundle "$BUNDLE" \
    --precision fp16 --views 2 --iters 100 --label "$M.ort-split"
```

For X-VLA, use the view count baked into the bundle. Confirm heavy graphs produced
`.engine` files; provider priority alone does not prove TensorRT executed the graph.

For a thermal run:

```bash
.venv-ort/bin/python -m bench ort-split --model "$M" --bundle "$BUNDLE" \
    --precision fp16 --views 2 --duration-s 300 --label "$M.ort-split.sustained"
```

Read latency drift and peak junction temperature together.

## 6. Check parity and generate the report

```bash
.venv-torch/bin/python -m bench parity results --reference "$M.torch-fp32"
.venv-torch/bin/python -m bench report results --out docs/RESULTS.md
```

A finite action is not enough. Check the signed cosine, maximum absolute difference,
percentage of action range, observation fingerprint, noise fingerprint, and bundle
identity. Commit result JSONs with the generated summary.

The complete supported matrix is:

```bash
MODEL=smolvla-base scripts/run_all.sh
MODEL=xvla-base scripts/run_all.sh
```

## Real observations

Synthetic inputs are suitable for timing and resource measurements. Use recorded frames
when action differences matter, and feed exactly the same frames to every backend:

```bash
python -m bench.tools.extract_frames --video <episode.mp4> \
    --out frames/ --count 30 --stride 10

python -m bench ort-split --model "$M" --bundle "$BUNDLE" \
    --obs frames:frames/ --label "$M.real-frames"
```

Camera capture, USB, control, and logging remain outside this harness and must be timed
in the robot process.
