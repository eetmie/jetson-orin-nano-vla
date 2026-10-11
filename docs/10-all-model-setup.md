# Benchmark all six models

Export portable bundles on a Linux Python 3.12 machine with enough RAM for the
stock policies. Then build and benchmark on an Orin Nano Super 8 GB with JetPack
7.2.1, CUDA 13 and TensorRT 10.16.2.10. Large policy exports run off the board;
the board needs neither PyTorch nor LeRobot.

## Export machine

```bash
git clone https://github.com/eetmie/jetson-orin-nano-vla.git
cd jetson-orin-nano-vla
export/setup.sh smolvla xvla groot groot17

bundle_root="$HOME/vla-bundles"
export/export.sh lerobot/smolvla_base "$bundle_root/smolvla-base" --views 2 --hoist-cross-kv
export/export.sh zuoxingdong/evo1_libero "$bundle_root/evo1-libero" --views 2
export/export.sh lerobot/xvla-base "$bundle_root/xvla-base" --views 3
export/export.sh nvidia/GR00T-N1.7-3B "$bundle_root/groot-n17-base"
export/export.sh nvidia/GR00T-N1.6-3B "$bundle_root/groot-n16-base"
export/export.sh pi05-libero "$bundle_root/pi05-libero"
```

Environment directories must be new. GR00T source is fetched with a sparse Git
checkout at a fixed commit. N1.7's auxiliary Cosmos tokenizer/processor is gated:
accept the model's terms and run `export/.venv-smolvla/bin/hf auth login` before
exporting. `--vlm-files /path/to/local/files` also works.

The pi0.5 command needs Docker with NVIDIA GPU support. It builds the pinned NGC
container, downloads checksum-pinned GCS checkpoint generations, converts to
PyTorch, and creates the compact 521-token bundle. It retains all model layers
and runs five full-policy fixture comparisons. Export stages and their logs are
in `pi05-libero.work/`. For an existing converted checkpoint, use
`export/pi05/export.sh OUT CONVERTED_CHECKPOINT`; its trained tensor content digest
must match the lock. The unused expert output head can differ after conversion;
the bundle records the actual checkpoint file hash.

Copy each **whole** bundle, including external ONNX weights, tokenizers,
fixtures and `MANIFEST.sha256`, into one directory on the board. Its subdirectories
must use the six model keys shown above. Engine files are built on the board.

## Orin

```bash
# In a fresh clone, after the documented JetPack/host setup:
scripts/12_env_trt.sh
scripts/14_env_triton.sh
python3 -m bench.tools.vla_setup --model all \
  --bundles-dir "$HOME/vla-bundles" --work-dir "$HOME/vla-runs/new-build" --keep-going
```

The work directory must be new. Models run serially so the 8 GB board only loads
one policy at a time. The workflow verifies bundle checksums, imports and GPU
access, builds ordinary engines and the selected AOT replacements, checks full
action fixtures, then runs each benchmark for 300 seconds. GR00T uses ordinary
native engines. `matrix.json` reports pass/fail; each model gets its own
`setup.json`, stage logs, doctor reports, caches and benchmark JSON.
`--keep-going` records failures and continues with the other models; the command
still exits nonzero if any model fails.

For a short functional test, add `--duration-s 10`. The three-call intermediate
baseline checks functionality; use the final duration run for speed/RAM readings.
To run one model, use `--model MODEL --bundle BUNDLE` instead of
`--model all --bundles-dir DIR`. `--dry-run` prints commands without importing GPU
packages. `--mode baseline` builds ordinary native engines. `--accumulate fp32`
is the default; pi0.5 always accumulates in FP32.

All native subprocesses run with Hub access disabled. The bundle supplies their
tokenizers and processors. Runtime and builder venvs link only JetPack's TensorRT
Python package; other system Python packages cannot satisfy missing imports.
AOT kernels are serialized into engines. The runtime venv has no Triton or Torch.

## Pins and evidence

`export/hub-lock.json` pins primary Hugging Face checkpoints and auxiliary models
to full commit SHAs. The exporter downloads through bounded HTTP ranges, checks
Hub snapshot identity and each file's LFS SHA-256 or Git-blob SHA-1, and preloads
pinned auxiliary snapshots before loading policies. Completed files are reused
only after verification; saved chunk hashes allow interrupted ranged files to
resume. Partial files without range metadata restart. Nested Hub reads
use the same lock. A custom remote
checkpoint requires an explicit full `--revision` SHA; local checkpoints are
accepted and the export manifest fixes the resulting artifact bytes.

`export/constraints-*.txt` and `requirements/*constraints.txt` fix transitive
Python dependencies as well as direct requirements. Bootstrap pip versions are
fixed. GR00T source commits are in `export/setup.sh`; pi0.5's source commit,
container digest and converted weight hash are in `export/pi05/source-lock.json`.
Its raw checkpoint object generations and SHA-256 hashes are in
`export/pi05/checkpoint-lock.json`. Downloading a moving `main` or tag is rejected.

JetPack's native CUDA/TensorRT libraries belong to the documented platform and
are recorded by doctor/benchmark reports. A fresh clone and venv test on an
existing board does not establish that a fresh OS image has every system library.
See the [test audit](09-test-audit.md) for actual coverage and run evidence.
