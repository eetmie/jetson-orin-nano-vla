# pyright: reportArgumentType=false
# (Triton's stubs type tl.constexpr params as `constexpr`; launches passing
#  a plain int misfire this check.)
"""Step-0 device validation. Run this BEFORE any benchmark work on a new box.

Standalone on purpose (no repo imports) so it can be copied to a device alone:

    python jetson_smoke.py

Checks, in order: torch imports and sees CUDA; triton imports and reports its
version; a trivial Triton kernel compiles, launches, and matches torch on the
GPU. Exit code 0 means the box is ready for bench/run.py.
"""

import sys


def main() -> int:
    print(f"python  : {sys.version.split()[0]}")

    try:
        import torch
    except ImportError as e:
        print(f"FAIL: torch import: {e}")
        return 1
    print(f"torch   : {torch.__version__} (cuda {torch.version.cuda})")

    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return 1
    cap = torch.cuda.get_device_capability(0)
    print(f"device  : {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")

    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        print(f"FAIL: triton import: {e}")
        return 1
    print(f"triton  : {triton.__version__}")

    @triton.jit
    def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        total = tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, total, mask=mask)

    n = 4099  # non-power-of-2: exercises masking
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    _add[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()

    if not torch.allclose(out, x + y):
        print("FAIL: triton kernel result mismatch vs torch")
        return 1
    print("OK: triton kernel compiled, ran on GPU, matches torch — ready to benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
