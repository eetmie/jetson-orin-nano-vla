# pyright: reportArgumentType=false, reportCallIssue=false
# (Triton's stubs type tl.constexpr params as `constexpr` and don't model
#  launch-time meta-parameters like num_warps; both checks misfire on every
#  kernel launch.)
"""Row-wise softmax over the last dimension of a 2-D tensor. Two variants.

`softmax` — one program per row, the entire row in a single block
(BLOCK_N = next_power_of_2(n_cols)). Simple and fast while a row fits.

`softmax_online` — for rows larger than one block. Streams each row in
TILE_N chunks keeping a running maximum `run_max` and a running sum
`run_sum` of exp(x - run_max); when a chunk raises the maximum, the
accumulated sum is rescaled by exp(old_max - new_max). A second pass over
the row normalizes and writes output. This is the online-softmax recurrence
that Flash-Attention builds on — the row never has to fit anywhere.

Numerical stability: both variants subtract the row max before
exponentiating, and all math runs in fp32 regardless of input dtype; the
store casts back to the output dtype.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    x_row_stride,
    out_row_stride,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    # Out-of-row lanes read -inf: they contribute exp(-inf) = 0 to the sum.
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=float("-inf"))
    x = x.to(tl.float32)

    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, numerator / denominator, mask=mask)


@triton.jit
def _softmax_online_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    x_row_stride,
    out_row_stride,
    TILE_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    x_row = x_ptr + row * x_row_stride

    run_max = tl.full((1,), float("-inf"), tl.float32)
    run_sum = tl.zeros((1,), tl.float32)

    # Pass 1: online max + rescaled sum, one tile at a time.
    for start in range(0, n_cols, TILE_N):
        cols = start + tl.arange(0, TILE_N)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        new_max = tl.maximum(run_max, tl.max(x, axis=0))
        run_sum = run_sum * tl.exp(run_max - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
        run_max = new_max

    # Pass 2: normalize and store.
    out_row = out_ptr + row * out_row_stride
    for start in range(0, n_cols, TILE_N):
        cols = start + tl.arange(0, TILE_N)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        tl.store(out_row + cols, tl.exp(x - run_max) / run_sum, mask=mask)


def _check_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D tensor, got shape {tuple(x.shape)}")
    return x.contiguous()


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Single-block row softmax. Requires the row to fit in one block."""
    x = _check_2d(x)
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_n = triton.next_power_of_2(n_cols)

    # Wider rows want more warps on the reduction; capped where returns flatten.
    num_warps = 4
    if block_n >= 2048:
        num_warps = 8
    if block_n >= 8192:
        num_warps = 16

    _softmax_kernel[(n_rows,)](
        x,
        out,
        n_cols,
        x.stride(0),
        out.stride(0),
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


def softmax_online(x: torch.Tensor, tile_n: int = 1024) -> torch.Tensor:
    """Streaming row softmax for rows of any length; two passes over the row."""
    x = _check_2d(x)
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    _softmax_online_kernel[(n_rows,)](
        x,
        out,
        n_cols,
        x.stride(0),
        out.stride(0),
        TILE_N=tile_n,
    )
    return out
