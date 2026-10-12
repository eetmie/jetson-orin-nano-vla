# 2. Python environments

For the seven supported native models, use the
[all-model setup workflow](10-all-model-setup.md). `.venv-trt` supplies TensorRT,
NumPy, OpenCV, tokenizers/processors and monitoring; `.venv-triton` adds the
SM87 AOT compiler. Both isolate Python dependencies and link only JetPack's
TensorRT package. Neither needs Torch. The reference/ORT environments below
remain available for optional comparisons; their constraints capture the existing
board environments and have not received the same fresh-install validation.

## Experimental uv setup for native environments

The native setup scripts default to the tested pip workflow. Set
`VLA_ENV_MANAGER=uv` to test the uv installer in fresh environment directories:

```bash
# Install the exact tool version recorded in .uv-version, once:
curl -LsSf https://astral.sh/uv/0.13.0/install.sh | sh

# In the repository root, on the Jetson:
VLA_ENV_MANAGER=uv scripts/12_env_trt.sh .venv-trt-uv
VLA_ENV_MANAGER=uv scripts/14_env_triton.sh .venv-triton-uv

python3 -m bench.tools.vla_setup --model smolvla-base \
  --bundle "$HOME/vla-bundles/smolvla-base" --work-dir "$HOME/vla-runs/uv-smoke" \
  --runtime-python .venv-trt-uv/bin/python --build-python .venv-triton-uv/bin/python \
  --duration-s 10
```

Open a new shell if `uv` is not yet on PATH, or set `UV_BIN` to its absolute path.
The scripts check the pinned uv version before creating an environment. They
refuse existing environment directories, use `/usr/bin/python3` from JetPack,
and link only the system TensorRT bindings. `VLA_SYSTEM_PYTHON` can select another
JetPack system interpreter explicitly; do not select a downloaded Python.

uv installs the complete role-specific dependency closure from
`requirements/trt-uv.lock.txt` or `requirements/triton-build-uv.lock.txt`, enforcing
SHA-256 hashes and requiring wheels. It does not resolve newer package versions
or compile Python dependencies on the board. The uv environments contain no pip;
use `uv pip check --python PATH/bin/python` to inspect them. Runtime and compiler
doctors run after installation, including an SM87 compilation probe for the builder.
Launch inference with the environment's Python; uv is needed only for setup.

The locks target Python 3.12 on Linux aarch64 and preserve the existing version
constraints. TensorRT/CUDA remain JetPack dependencies outside the Python locks.
Generate updated locks deliberately with the version from `.uv-version`:

```bash
uv --no-config pip compile requirements/trt.txt \
  --python-version 3.12 --python-platform aarch64-unknown-linux-gnu \
  --only-binary :all: --generate-hashes --no-annotate \
  --output-file requirements/trt-uv.lock.txt
uv --no-config pip compile requirements/triton-build.txt \
  --python-version 3.12 --python-platform aarch64-unknown-linux-gnu \
  --only-binary :all: --generate-hashes --no-annotate \
  --output-file requirements/triton-build-uv.lock.txt
```

Use a cache on the same filesystem as the environments so uv can hardlink shared
package files. `UV_CACHE_DIR` selects a separate experiment cache if desired.
Keep verified engine caches, bundles, and results while comparing installers.
Unused venvs consume disk space; they do not consume inference RAM.

The [2026-10-12 experiment](09-test-audit.md#uv-native-environment-experiment--2026-10-12)
checks repeated installation, all-model doctors and a fresh SmolVLA build/inference
smoke test. The subsequent [second-board clean install](09-test-audit.md#fresh-second-board-clean-install--2026-10-12)
also passes fresh exports, builds and 300-second benchmarks for all seven models.
The pinned uv path is validated for this native stack; it remains opt-in while
reference environments and exporters retain their existing pip workflows.

Use uv as the installer and keep the separate runtime/compiler venvs on bare
metal. A venv adds no resident process, and switching installers does not itself
lower inference RAM. The lean runtime saves memory by excluding Torch, ONNX Runtime
and the Triton compiler. Keep heavy exports on the export host; the Jetson needs
no Docker for this workflow.

## Optional reference and ORT environments

| venv | backend | brings |
|---|---|---|
| `.venv-torch` | `torch` (SmolVLA) | lerobot **0.5.1**, JetPack-matched torch |
| `.venv-torch-xvla` | `torch` (X-VLA) | lerobot **0.6.1** + `[xvla]` |
| `.venv-ort` | `ort-split`, `trt-split` (every family) | onnxruntime-gpu 1.24 + system TensorRT (`trt-split` does not use onnxruntime) |

```bash
scripts/10_env_torch.sh        # -> .venv-torch
scripts/13_env_torch_xvla.sh   # -> .venv-torch-xvla
scripts/11_env_ort.sh          # -> .venv-ort
```

Two lerobot versions cannot share a venv, and the split is not about a feature list:
the X-VLA export tooling only works against the layout 0.6.1 ships (see
`docs/03-backends.md`), while the SmolVLA side stays on the version its checkpoint was
trained and exported with. Do not trust the version string to tell you which layout you
have — an install reporting `0.5.1` on this machine carries an `xvla` policy with a
*vendored* Florence2, which is not what 0.6.1 has. Check the module, not the number.
The EVO1 native fixture was emitted with LeRobot 0.6.1. All **ONNX** backends
are version-agnostic at runtime; they never import lerobot.

They are separate because all three want to own torch and onnxruntime, and on this
board those are exactly the two wheels that are painful to reinstall. One shared venv
means every `pip install` is a coin flip over which backend still works afterwards.

## The one trap that produces a wrong number instead of an error

`pip install lerobot` resolves `torch` from PyPI, which will happily replace the
JetPack-matched wheel you installed. Whether the replacement works on this board is not
something to find out from a latency number: a torch that cannot reach the iGPU falls
back to CPU silently — no error, no warning, just a figure roughly twenty times too
slow, which would make every other backend look far better than it is.

(Do not assume the PyPI aarch64 wheel is CPU-only. It is not, at least off-Jetson:
`torch 2.11.0+cu130` installed from plain PyPI on aarch64 reports
`torch.cuda.is_available() == True`. What matters is that the wheel matches this
board's CUDA stack, not which index it came from.)

So the order in `scripts/10_env_torch.sh` is: lerobot first, then force the
JetPack-matched torch on top, then assert:

```python
assert torch.cuda.is_available()
```

The script fails loudly if that assert does not hold. Do not skip it — it is the only
thing standing between a resolver decision and a wrong number.

## Where the aarch64 CUDA-13 wheels live

There is **no `jp7` index** on Jetson AI Lab. The CUDA 13 aarch64 builds are under
`sbsa`:

```
https://pypi.jetson-ai-lab.io/sbsa/cu130
```

Verified from there: `onnxruntime-gpu==1.24.0` (cp312) registering all three EPs, and
torch/torchvision for CUDA 13. A benign `GPU device discovery failed
.../card1/device/vendor` warning on ORT import is a Jetson iGPU DRM-probe quirk — the
CUDA EP works regardless.

## `--system-site-packages` for the ORT venv

`tensorrt` ships with JetPack and is **not** pip-installable. A clean venv cannot see
it, and ORT's TensorRT EP will then refuse to register — silently downgrading the
benchmark to the CUDA EP. `scripts/11_env_ort.sh` creates the venv with
`--system-site-packages` and asserts `TensorrtExecutionProvider` is in
`get_available_providers()` before declaring success.

## The X-VLA venv has a second trap: scipy shadowing

With `--system-site-packages`, the JetPack `scipy` (built against numpy 1.x) shadows
through and makes `import lerobot` fail on `cannot import name 'Inf' from 'numpy'`. pip
then "resolves" it by downgrading numpy, which breaks lerobot's own `numpy>=2` pin.
Installing both **into** the venv settles it:

```bash
pip install -c requirements/torch-xvla-constraints.txt --ignore-installed \
    "numpy==2.5.2" "scipy==1.18.1"
```

`scripts/13_env_torch_xvla.sh` does this and then asserts that `XVLAPolicy` imports.

## Version pins that matter

SmolVLA uses LeRobot 0.5.1; X-VLA and the EVO1 export workflow use LeRobot 0.6.1
because those are the layouts their verified base exports target. Tokenizer identity and
sequence length are part of
each bundle contract; changing either means the runtimes are no longer comparable.
