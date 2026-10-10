"""EVO1 (InternViT) vision attention for SM87 on the separate Q/K/V projections.

The exported block computes q/k/v = x @ W + b ([B,S,H*D] each), HALF QK^T, a HALF
multiply by the scale, an FP32 softmax rounded to HALF and HALF P @ V, no mask. Here:
HALF score rounding, then the scale multiply rounded to HALF (the scale is exact in
FP32, so FP32-multiply-then-round equals the HALF multiply), FP32 online softmax and
accumulation; unnormalized probabilities round to HALF before PV (approximate;
full-action gates required). S need not be a tile multiple.
"""
import triton
import triton.language as tl


@triton.jit
def evo1_vision_attention_aot(Q, K, V, O, S: tl.constexpr, H: tl.constexpr, SCALE: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    batch = tl.program_id(2)
    head = tl.program_id(1)
    stride = H*D
    base = batch*S*stride + head*D
    rows = tl.program_id(0)*BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    keys = tl.arange(0, BLOCK_N)
    valid = rows < S
    q = tl.load(Q+base+rows[:, None]*stride+dims[None, :], mask=valid[:, None], other=0)
    maximum = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accum = tl.zeros((BLOCK_M, D), tl.float32)
    for start in range(0, S, BLOCK_N):
        cols = start+keys
        inside = cols < S
        kt = tl.load(K+base+cols[None, :]*stride+dims[:, None], mask=inside[None, :], other=0)
        score = tl.dot(q, kt).to(tl.float16).to(tl.float32)
        score = (score*SCALE).to(tl.float16).to(tl.float32)
        if S % BLOCK_N != 0:
            score = tl.where(inside[None, :], score, float('-inf'))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        weight = tl.exp(score-next_max[:, None])
        alpha = tl.exp(maximum-next_max)
        denominator = denominator*alpha+tl.sum(weight, 1)
        accum = accum*alpha[:, None]
        value = tl.load(V+base+cols[:, None]*stride+dims[None, :], mask=inside[:, None], other=0)
        accum = tl.dot(weight.to(tl.float16), value, accum)
        maximum = next_max
    out = accum/denominator[:, None]
    tl.store(O+base+rows[:, None]*stride+dims[None, :], out, mask=valid[:, None])


@triton.jit
def gemv_bias_aot(X, W, B, Y, N: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    """y[1,N] = x[1,K] @ W[N,K]^T + b with FP32 accumulation, HALF in and out.

    EVO1's action_output pool layer (K=44800): bandwidth-bound, so accumulating in
    FP32 instead of TensorRT's FP16-accumulating tensor-core tactic costs nothing.
    """
    rows = tl.program_id(0)*BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for start in range(0, K, BLOCK_K):
        inside = start+cols < K
        x = tl.load(X+start+cols, mask=inside, other=0).to(tl.float32)
        w = tl.load(W+rows[:, None]*K+start+cols[None, :], mask=inside[None, :], other=0).to(tl.float32)
        acc += tl.sum(w*x[None, :], 1)
    acc += tl.load(B+rows).to(tl.float32)
    tl.store(Y+rows, acc.to(tl.float16))
