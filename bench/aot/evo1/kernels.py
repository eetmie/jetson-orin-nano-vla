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


@triton.jit
def step_attention_aot(Q, K, V, Mask, O, NQ: tl.constexpr, NK: tl.constexpr, H: tl.constexpr,
                       D: tl.constexpr, KD: tl.constexpr, SCALE: tl.constexpr,
                       SQH: tl.constexpr, SQS: tl.constexpr, SKH: tl.constexpr, SKS: tl.constexpr,
                       SVH: tl.constexpr, SVS: tl.constexpr, SCORE_FP32: tl.constexpr,
                       TWO_PASS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """EVO1 action-step cross-attention on the cached K/V; O [1,NQ,H*D].

    The graph transposes the cached key to [H,D,NK] every step, takes HALF QK^T, a HALF
    multiply by SCALE, adds the HALF key mask [NK] and runs an FP32 softmax rounded to
    HALF before PV. Here Q, K and V are read in place (head/row strides per tensor, D=112
    padded to KD lanes with zeros). SCORE_FP32 keeps QK^T*SCALE+mask in FP32 instead of
    the two HALF roundings; TWO_PASS rounds normalized probabilities as the graph does.
    """
    head = tl.program_id(1)
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    dims = tl.arange(0, KD)
    keys = tl.arange(0, BN)
    q = tl.load(Q + head*SQH + rows[:, None]*SQS + dims[None, :],
                mask=(rows[:, None] < NQ) & (dims[None, :] < D), other=0)
    kb = K + head*SKH
    vb = V + head*SVH
    maximum = tl.full((BM,), float('-inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accum = tl.zeros((BM, KD), tl.float32)
    for start in range(0, NK, BN):
        cols = start + keys
        k = tl.load(kb + cols[None, :]*SKS + dims[:, None], mask=(cols[None, :] < NK) & (dims[:, None] < D), other=0)
        m = tl.load(Mask + cols, mask=cols < NK, other=0).to(tl.float32)
        if SCORE_FP32:
            score = tl.dot(q, k)*SCALE + m[None, :]
        else:
            score = (tl.dot(q, k).to(tl.float16).to(tl.float32)*SCALE).to(tl.float16).to(tl.float32)
            score = (score + m[None, :]).to(tl.float16).to(tl.float32)
        score = tl.where(cols[None, :] < NK, score, float('-inf'))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        weight = tl.exp(score - next_max[:, None])
        alpha = tl.exp(maximum - next_max)
        denominator = denominator*alpha + tl.sum(weight, 1)
        if not TWO_PASS:
            v = tl.load(vb + cols[:, None]*SVS + dims[None, :], mask=(cols[:, None] < NK) & (dims[None, :] < D), other=0)
            accum = tl.dot(weight.to(tl.float16), v, accum*alpha[:, None])
        maximum = next_max
    if TWO_PASS:
        for start in range(0, NK, BN):
            cols = start + keys
            k = tl.load(kb + cols[None, :]*SKS + dims[:, None], mask=(cols[None, :] < NK) & (dims[:, None] < D), other=0)
            m = tl.load(Mask + cols, mask=cols < NK, other=0).to(tl.float32)
            if SCORE_FP32:
                score = tl.dot(q, k)*SCALE + m[None, :]
            else:
                score = (tl.dot(q, k).to(tl.float16).to(tl.float32)*SCALE).to(tl.float16).to(tl.float32)
                score = (score + m[None, :]).to(tl.float16).to(tl.float32)
            score = tl.where(cols[None, :] < NK, score, float('-inf'))
            p = (tl.exp(score - maximum[:, None])/denominator[:, None]).to(tl.float16)
            v = tl.load(vb + cols[:, None]*SVS + dims[None, :], mask=(cols[:, None] < NK) & (dims[None, :] < D), other=0)
            accum = tl.dot(p, v, accum)
        result = accum
    else:
        result = accum/denominator[:, None]
    tl.store(O + rows[:, None]*(H*D) + head*D + dims[None, :], result.to(tl.float16),
             mask=(rows[:, None] < NQ) & (dims[None, :] < D))
