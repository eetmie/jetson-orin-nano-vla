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


@triton.jit
def masked_softmax_aot(X, Mask, n_cols, row_stride, Y, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    cols=tl.arange(0,BLOCK_N)
    valid=cols<n_cols
    x=tl.load(X+row*row_stride+cols,mask=valid,other=0).to(tl.float32)
    mask=tl.load(Mask+(row%n_cols)*n_cols+cols,mask=valid,other=0).to(tl.float32)
    # Original ONNX Add is FP16, followed by Cast -> FP32 Softmax -> Cast FP16.
    x=(x+mask).to(tl.float16).to(tl.float32)
    x=tl.where(valid,x,float('-inf'))
    numerator=tl.exp(x-tl.max(x,0))
    prob=(numerator/tl.sum(numerator,0)).to(tl.float16)
    # Preserve the model's following IsNaN/Where(zero, probability) guard.
    prob=tl.where(prob==prob,prob,0.0)
    tl.store(Y+row*row_stride+cols,prob,mask=valid)
