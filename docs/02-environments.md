# 2. Environments — three venvs, and why they cannot be one

| venv | backend | brings |
|---|---|---|
| `.venv-torch` | `torch` | lerobot 0.5.1, JetPack-matched torch |
| `.venv-ort` | `ort-split` | onnxruntime-gpu 1.24 + system TensorRT |
| `.venv-tether` | `tether` | fastcrest-tether and its own torch/ORT pins |

```bash
scripts/10_env_torch.sh      # -> .venv-torch
scripts/11_env_ort.sh        # -> .venv-ort
scripts/12_env_tether.sh     # -> .venv-tether
```

They are separate because all three want to own torch and onnxruntime, and on this
board those are exactly the two wheels that are painful to reinstall. One shared venv
means every `pip install` is a coin flip over which backend still works afterwards.

## The one trap that produces a wrong number instead of an error

`pip install lerobot` resolves `torch` from PyPI. **The aarch64 PyPI wheel is
CPU-only.** Install it after the JetPack-matched wheel and the PyTorch "GPU baseline"
quietly becomes a CPU run — no error, no warning, just a number roughly twenty times
too slow, which would make every other backend look far better than it is.

So the order in `scripts/10_env_torch.sh` is: lerobot first, then force the CUDA-13
aarch64 torch on top, then assert:

```python
assert torch.cuda.is_available()
```

The script fails loudly if that assert does not hold. Do not skip it.

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

## Version pins that matter

`transformers==5.3.0` and `lerobot==0.5.1` match what the DGX Spark trained and
exported with. The tokenizer especially: a mismatched tokenizer means different token
ids, which means a different language embedding, which means the policy is
conditioned on something it was never fine-tuned on. The harness sidesteps most of
this by loading the tokenizer *from the export bundle* for both the torch and the ORT
backend, so the two are provably fed the same tokens.
