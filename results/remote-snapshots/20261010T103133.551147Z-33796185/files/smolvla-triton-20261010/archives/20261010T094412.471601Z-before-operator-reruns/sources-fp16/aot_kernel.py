"""Lab row-softmax adapted to TensorRT's input/scalar/output kernel ABI.

Algorithm adapted from vendor/lab_softmax.py. MIT Copyright (c) 2026 Jeremy
Gracey; retain vendor/LICENSE.triton-kernel-lab. Accumulation stays FP32.
"""
import triton
import triton.language as tl


@triton.jit
def softmax_aot(X, n_cols, row_stride, Y, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    cols=tl.arange(0,BLOCK_N)
    valid=cols<n_cols
    x=tl.load(X+row*row_stride+cols,mask=valid,other=float('-inf')).to(tl.float32)
    numer=tl.exp(x-tl.max(x,0))
    tl.store(Y+row*row_stride+cols,numer/tl.sum(numer,0),mask=valid)
