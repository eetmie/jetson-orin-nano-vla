# WAM benchmark probes

Both probes use two synthetic cameras and the fixed instruction “Pick up the cube.” From the repository root, set `export PYTHONPATH="$PWD:$PYTHONPATH"`, `REPO="$PWD"` and `WAM="$PWD/scripts/wam"` on each machine. Export in a separate working directory per model with the checkpoint snapshot in `ckpt/`. The manifests alongside the scripts pin the checkpoint and LeRobot revisions. Parity is measured against the stock policy run in FP32, with the same noise.

## Cosmos3-Edge LIBERO

On a CUDA host with the pinned LeRobot policy, PyTorch and ONNX installed, generate the portable bundle and the stock FP32/BF16 reference chunks. Copy `pack/` to the Jetson.

```sh
python "$WAM/cosmos3_edge/fixtures.py"
python "$WAM/cosmos3_edge/export_full.py"
python "$WAM/cosmos3_edge/reference.py"
```

On the Jetson, build the engines sequentially with TensorRT and run the five-minute benchmark. For the three-call Spark smoke test, use `spark` in place of `jetson` inside a GPU container with a 6.5 GB memory limit. Each build stops when the board passes 7.3 GB in use.

```sh
python3 "$WAM/cosmos3_edge/build_pack.py" pack jetson
OPENBLAS_NUM_THREADS=1 python3 "$WAM/cosmos3_edge/build_pack.py" pack jetson run
```

## LaWAM LIBERO

On the export host, export FP32 graphs: the front end, the native DINO encoder and the 8-token flow sampler. Then generate the reference fixtures. The stock policy runs its VLM in BF16, which drifts as far as cos 0.9997 from FP32, so the engines run FP16 and are held to the FP32 reference.

```sh
python "$WAM/lawam/export_front.py" --fp32
python "$WAM/lawam/dino_native.py" --fp32
python "$WAM/lawam/export_sampler_full.py"
for attention in eager sdpa fp32; do python "$WAM/lawam/extra_refs.py" --attention $attention; done
cp front32/vis*.onnx front32/lang*.onnx front32/latent.onnx front32/input.npz full/
cp dino-fp32.onnx full/dino.onnx
```

Convert them to mixed FP16 (norms, softmax and the front end's residual stream stay FP32), and make every FP16 MatMul accumulate in FP32. Run this from `export/` with the X-VLA venv, which has ONNX Runtime.

```sh
FRONT="vis0.onnx vis1.onnx vis2.onnx vis3.onnx lang0.onnx lang1.onnx lang2.onnx lang3.onnx dino.onnx latent.onnx"
FLOW="kv.onnx g0.onnx g1.onnx g2.onnx g3.onnx"
cd "$REPO/export"
.venv-xvla/bin/python -m vla_common.fp16_mixed --bundle "$OLDPWD/full" --graphs $FRONT --fp32-residual
.venv-xvla/bin/python -m vla_common.fp16_mixed --bundle "$OLDPWD/full" --graphs $FLOW --half-io '.*'
.venv-xvla/bin/python -m vla_common.fp32_accumulate --bundle "$OLDPWD/full" --graphs $FRONT $FLOW
```

Copy `full/` to the Jetson, build each graph under the same memory guard, and run the five-minute benchmark.

```sh
python3 "$WAM/cosmos3_edge/build_pack.py" full jetson --opt-level 3 --workspace 256 \
    vis0 vis1 vis2 vis3 lang0 lang1 lang2 lang3 dino latent kv g0 g1 g2 g3
python3 "$WAM/lawam/lawam_jetson_benchmark.py" full
```
