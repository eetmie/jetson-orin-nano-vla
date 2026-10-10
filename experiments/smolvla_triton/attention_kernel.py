"""Experimental SM87 tiled attention using the online-softmax recurrence.

Fixed SmolVLA vision layout: scaled HALF Q[B,H,S,64], scaled HALF Kt[B,H,64,S],
HALF V[B,H,S,64], shared HALF mask[B,1,S,S]. FP32 dot/normalization accumulation.
Scores round to HALF before and after mask addition, as the exported graph does.
Online PV rounds *unnormalized* probabilities to HALF, so this is an approximate
replacement; full-action gates are required. It never writes an S x S matrix.
"""
import triton
import triton.language as tl


@triton.jit
def attention_aot(Q, Kt, V, Mask, n_ctx, O,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    head = tl.program_id(1)
    rows = tl.program_id(0)*BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    keys = tl.arange(0, BLOCK_N)
    base = head*n_ctx*D
    q = tl.load(Q+base+rows[:, None]*D+dims[None, :], mask=rows[:, None]<n_ctx, other=0)
    maximum = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accum = tl.zeros((BLOCK_M, D), tl.float32)
    bad = tl.full((BLOCK_M,), False, tl.int1)
    for start in range(0, n_ctx, BLOCK_N):
        cols = start+keys
        kt = tl.load(Kt+base+dims[:, None]*n_ctx+cols[None, :],
                     mask=cols[None, :]<n_ctx, other=0)
        score = tl.dot(q, kt).to(tl.float16).to(tl.float32)
        bias = tl.load(Mask+rows[:, None]*n_ctx+cols[None, :],
                       mask=(rows[:, None]<n_ctx)&(cols[None, :]<n_ctx), other=float('-inf'))
        score = (score+bias.to(tl.float32)).to(tl.float16).to(tl.float32)
        bad = bad | (tl.sum(((score!=score)|(score==float('inf'))).to(tl.int32), 1)>0)
        next_max = tl.maximum(maximum, tl.max(score, 1))
        shift = tl.where(next_max==float('-inf'), 0.0, next_max)
        weight = tl.exp(score-shift[:, None])
        alpha = tl.exp(maximum-shift)
        denominator = denominator*alpha+tl.sum(weight, 1)
        accum = accum*alpha[:, None]
        value = tl.load(V+base+cols[:, None]*D+dims[None, :], mask=cols[:, None]<n_ctx, other=0)
        accum = tl.dot(weight.to(tl.float16), value, accum)
        maximum = next_max
    out = accum/tl.where(denominator>0, denominator, 1.0)[:, None]
    out = tl.where(bad[:, None], 0.0, out)
    tl.store(O+base+rows[:, None]*D+dims[None, :], out, mask=rows[:, None]<n_ctx)
