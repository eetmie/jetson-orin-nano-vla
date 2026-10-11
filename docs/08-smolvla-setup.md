# Build the selected SmolVLA runtime

This recipe builds the mixed-FP16, FP32-accumulating SmolVLA configuration in the
README. It takes a complete portable bundle and builds baseline, fused vision and
packed expert engines on the Jetson, then runs the stock-policy and device-chain
fixture gates. It never needs a dated experiment cache. Keep the bundle and the
resulting `setup.json`, doctor reports and logs together.

## Export machine

Clone this repository on a Linux Python 3.12 machine with enough RAM for the stock
policy and ONNX export. SmolVLA export can run on the CPU. Pin the checkpoint for
repeatable downloads; the revision below is the one in the measured checkpoint's
local Hugging Face download metadata.

```bash
git clone https://github.com/eetmie/jetson-orin-nano-vla.git
cd jetson-orin-nano-vla
export/setup.sh smolvla
export/export.sh lerobot/smolvla_base "$HOME/bundles/smolvla-optimized" \
  --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 --views 2 --hoist-cross-kv
```

`--revision` is forwarded to the checksum-verifying Hub downloader and uses a
separate checkpoint directory. Required auxiliary snapshots are pinned too.
Only full commit SHAs are accepted; known public checkpoints default to the source lock. For your own local
checkpoint, replace the HF id with its path and omit `--revision`.
[HF documents revision selection](https://huggingface.co/docs/huggingface_hub/guides/cli#download-a-specific-revision).
Copy the **entire** bundle to the board, including `.onnx.data`, tokenizer, stats,
embedding table, fixture and `MANIFEST.sha256`. The older HF ORT-only bundles lack
the native reference fixture and are rejected by this workflow.

## Jetson

Tested platform: Orin Nano Super SM87, JetPack 7.2.1, CUDA 13, Python 3.12 and
TensorRT 10.16.2.10. Complete the [host setup](01-host-setup.md) first; the commands
below install Python packages into new environments, leaving existing environments
alone. JetPack's TensorRT and CUDA remain system dependencies.
Environment directories must be new. Each venv links only JetPack's `tensorrt`
package and excludes unrelated system Python packages; `pip check` must pass.
The native pip dependencies are constrained to the isolated validation versions in
`requirements/trt-smolvla-constraints.txt`; the builder additionally pins Triton 3.6.0.

```bash
# from the cloned repository on the Jetson
scripts/12_env_trt.sh                 # .venv-trt: native runtime
scripts/14_env_triton.sh              # .venv-triton: AOT builder
python3 -m bench.tools.smolvla_setup \
  --bundle "$HOME/bundles/smolvla-optimized" \
  --work-dir "$HOME/vla-builds/smolvla-$(date -u +%Y%m%dT%H%M%SZ)"
```

The work directory must be new and outside the bundle. Stages execute serially;
each writes its own log. A failure stops the workflow, reports the relevant log,
and leaves artifacts available for inspection. After fixing it, choose a new work
directory. `--dry-run` prints commands without importing GPU packages or creating
files. Override `--runtime-python` and `--build-python` for environments elsewhere.
`--duration-s 10` gives a short inference smoke test; the default is 300 seconds.
The three-call baseline checks functionality, rather than supplying a comparable
speed/RAM measurement. `--accumulate auto` selects the older accumulation choice;
the default is `fp32` throughout both builds.

The selected configuration requires `--hoist-cross-kv` at export for the reported
speed. Graph matching and native fixture gates run on the actual exported graphs;
fine-tuned weights need their own validation. Preserve the camera resolution,
two views and ten denoising steps for comparisons with the README.

For subsequent runs, set `smol_work` to the successful work directory:

```bash
smol_work="$HOME/vla-builds/your-successful-build"
.venv-trt/bin/python experiments/smolvla_triton/run_candidate.py trt-split \
  --model smolvla-base --bundle "$HOME/bundles/smolvla-optimized" \
  --cache-dir "$smol_work/optimized-cache" --accumulate fp32 --chain graph --views 2 \
  --lean-tokenizer --trim-host-heap --warmup 10 --idle-s 3 --duration-s 300 \
  --out "results/smolvla-$(date -u +%Y%m%dT%H%M%SZ).json"
```

Use the candidate wrapper: it verifies engine and ONNX hashes before loading the
patched engines. The ordinary `bench trt-split` path rebuilds from unmodified ONNX.
AOT kernels are serialized into the engines; inference needs neither Triton nor
Torch. The lean tokenizer also avoids importing Transformers and bounds the prompt
cache to eight entries. The new native environments omit ONNX Runtime and LeRobot.
The ordinary CLI can also use `.venv-trt` (which supplies the
Transformers tokenizer for the other model families). Reference/operator checks needing Torch still use their
separate PyTorch environment; see the [experiment README](../experiments/smolvla_triton/README.md).

## What a clean setup proves

`python -m bench.tools.doctor --role runtime` checks real imports and a CUDA
allocation/free. `--role build` additionally checks Triton and the actual TensorRT
AOT plugin registrations and compiles an SM87 PTX probe. The explicit SM87 target
avoids Torch-dependent automatic driver discovery. Full compilation and inference add checks that imports
cannot provide. TensorRT's [AOT guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/pluginGuide.html)
describes the compiler/runtime separation.

Fresh venvs, source directories and engine caches on this board are enough to
find Python dependency and first-build failures. Linking JetPack's TensorRT
bindings still does **not** certify a clean operating system. A clean flash or
matching container adds coverage for accidental system
packages and library paths. Internet documentation helps verify requirements; it
cannot establish that CUDA libraries and compiled kernels execute on SM87.
See the [test audit](09-test-audit.md) for coverage and remaining checks.
