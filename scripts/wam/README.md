# WAM benchmark probes

Both probes use two synthetic cameras and the fixed instruction “Pick up the cube.” From the repository root, set `export PYTHONPATH="$PWD:$PYTHONPATH"` and `WAM="$PWD/scripts/wam"` on each machine. Export in a separate working directory per model with the checkpoint snapshot in `ckpt/`. The manifests alongside the scripts pin the checkpoint and LeRobot revisions.

## Cosmos3-Edge LIBERO

On a CUDA host with the pinned LeRobot policy, PyTorch and ONNX installed, generate the portable bundle. Copy `pack/` to the Jetson.

```sh
python "$WAM/cosmos3_edge/fixtures.py"
python "$WAM/cosmos3_edge/export_full.py"
```

On the Jetson, build the engines sequentially with TensorRT and run the five-minute benchmark. For the three-call Spark smoke test, use `spark` in place of `jetson` inside a GPU container with a 6.5 GB memory limit.

```sh
python3 "$WAM/cosmos3_edge/build_pack.py" pack jetson
OPENBLAS_NUM_THREADS=1 python3 "$WAM/cosmos3_edge/build_pack.py" pack jetson run
```

## LaWAM LIBERO

On the export host, generate the front end, native DINO encoder, flow sampler and reference fixtures. Copy `full/` to the Jetson.

```sh
python "$WAM/lawam/export_front.py"
python "$WAM/lawam/dino_native.py"
python "$WAM/lawam/export_sampler_full.py"
python "$WAM/lawam/extra_refs.py" --attention eager
python "$WAM/lawam/extra_refs.py" --attention sdpa
cp front/vis*.onnx front/lang*.onnx front/latent.onnx front/input.npz full/
cp dino-native.onnx full/dino.onnx
```

On the Jetson, build each graph and run the five-minute benchmark.

```sh
for graph in full/*.onnx; do
    trtexec --stronglyTyped --onnx="$graph" --saveEngine="${graph%.onnx}.engine" \
        --memPoolSize=workspace:64 --builderOptimizationLevel=1 --skipInference || exit 1
done
python3 "$WAM/lawam/lawam_jetson_benchmark.py" full
```
