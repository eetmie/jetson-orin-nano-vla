# 2. Environments — three venvs, and why they cannot be one

| venv | backend | brings |
|---|---|---|
| `.venv-torch` | `torch` (SmolVLA) | lerobot **0.5.1**, JetPack-matched torch |
| `.venv-torch-xvla` | `torch` (X-VLA) | lerobot **0.6.1** + `[xvla]` |
| `.venv-ort` | `ort-split` | onnxruntime-gpu 1.24 + system TensorRT |

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
Only the **ONNX** backends are version-agnostic; they never import lerobot.

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
pip install --ignore-installed "numpy>=2.2.6" "scipy>=1.14"
```

`scripts/13_env_torch_xvla.sh` does this and then asserts that `XVLAPolicy` imports.

## Version pins that matter

`transformers==5.3.0` and `lerobot==0.5.1` match what the DGX Spark trained and
exported with. The tokenizer especially: a mismatched tokenizer means different token
ids, which means a different language embedding, which means the policy is
conditioned on something it was never fine-tuned on. The harness sidesteps most of
this by loading the tokenizer *from the export bundle* for both the torch and the ORT
backend, so the two are provably fed the same tokens.
