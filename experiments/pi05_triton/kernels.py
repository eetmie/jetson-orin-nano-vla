"""π0.5 (Gemma) multi-query attention for SM87: 8 query heads share one K/V head, D=256.

The exported layer computes, for each head, s = HALF(q·k) with FP32 accumulation, a
HALF multiply by 1/16 (exact), FP32 logits + mask, an FP32 softmax rounded to HALF, and
HALF(P·V) with FP32 accumulation. Two passes over the keys reproduce that order: the
first finds each row's max and denominator, the second rounds the normalized
probabilities to HALF before PV, as the graph does (only the denominator's summation
order differs). Rows of all heads read the same K/V tiles.
"""
import triton
import triton.language as tl


@triton.jit
def mqa_attention_aot(Q, K, V, Mask, O, N_Q: tl.constexpr, N_KV: tl.constexpr, H: tl.constexpr,
                      D: tl.constexpr, SCALE: tl.constexpr, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr):
    # Q [H, N_Q, D] HALF; K, V [N_KV, D] HALF; Mask [N_Q, N_KV] FP32; O [N_Q, H*D] HALF.
    head = tl.program_id(1)
    rows = tl.program_id(0)*BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    keys = tl.arange(0, BLOCK_N)
    valid = rows < N_Q
    q = tl.load(Q + head*N_Q*D + rows[:, None]*D + dims[None, :], mask=valid[:, None], other=0)
    maximum = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    for start in range(0, N_KV, BLOCK_N):
        cols = start + keys
        inside = cols < N_KV
        kt = tl.load(K + cols[None, :]*D + dims[:, None], mask=inside[None, :], other=0)
        s = (tl.dot(q, kt).to(tl.float16).to(tl.float32)*SCALE).to(tl.float16).to(tl.float32)
        logit = s + tl.load(Mask + rows[:, None]*N_KV + cols[None, :],
                            mask=valid[:, None] & inside[None, :], other=0)
        logit = tl.where(inside[None, :], logit, float('-inf'))
        new_max = tl.maximum(maximum, tl.max(logit, 1))
        denominator = denominator*tl.exp(maximum - new_max) + tl.sum(tl.exp(logit - new_max[:, None]), 1)
        maximum = new_max
    accum = tl.zeros((BLOCK_M, D), tl.float32)
    for start in range(0, N_KV, BLOCK_N):
        cols = start + keys
        inside = cols < N_KV
        kt = tl.load(K + cols[None, :]*D + dims[:, None], mask=inside[None, :], other=0)
        s = (tl.dot(q, kt).to(tl.float16).to(tl.float32)*SCALE).to(tl.float16).to(tl.float32)
        logit = s + tl.load(Mask + rows[:, None]*N_KV + cols[None, :],
                            mask=valid[:, None] & inside[None, :], other=0)
        logit = tl.where(inside[None, :], logit, float('-inf'))
        p = (tl.exp(logit - maximum[:, None])/denominator[:, None]).to(tl.float16)
        v = tl.load(V + cols[:, None]*D + dims[None, :], mask=inside[:, None], other=0)
        accum = tl.dot(p, v, accum)
    tl.store(O + rows[:, None]*(H*D) + head*D + dims[None, :], accum.to(tl.float16), mask=valid[:, None])


@triton.jit
def _h(x):
    """Round to HALF and back: one FP16 op of the exported graph."""
    return x.to(tl.float16).to(tl.float32)


@triton.jit
def gated_mlp_aot(X, Wg, Wu, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Y = gelu_tanh(HALF(X@Wg)) * HALF(X@Wu), both GEMMs FP32-accumulated, side by side.

    X [M,K], Wg/Wu [K,N] HALF. The epilogue rounds to HALF after every op, in the
    exported graph's order (g*g, *g, *0.0447, +g, *0.7979, tanh, +1, *g, *0.5, *u).
    """
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm*BM + tl.arange(0, BM)
    rn = pn*BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    g = tl.zeros((BM, BN), tl.float32)
    u = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        x = tl.load(X + rm[:, None]*K + (k+rk)[None, :], mask=rm[:, None] < M, other=0)
        g = tl.dot(x, tl.load(Wg + (k+rk)[:, None]*N + rn[None, :]), g)
        u = tl.dot(x, tl.load(Wu + (k+rk)[:, None]*N + rn[None, :]), u)
    g = _h(g)
    u = _h(u)
    t = _h(g*g)
    t = _h(g*t)
    t = _h(0.044708251953125*t)
    t = _h(g + t)
    t = _h(0.7978515625*t)
    e2 = tl.exp(2.0*t)
    t = _h(1.0 - 2.0/(e2 + 1.0))          # tanh
    t = _h(1.0 + t)
    t = _h(g*t)
    t = _h(0.5*t)
    tl.store(Y + rm[:, None]*N + rn[None, :], (t*u).to(tl.float16), mask=rm[:, None] < M)


@triton.jit
def mqa_attention_online_aot(Q, K, V, Mask, O, N_Q: tl.constexpr, N_KV: tl.constexpr, H: tl.constexpr,
                             D: tl.constexpr, SCALE: tl.constexpr, BLOCK_M: tl.constexpr,
                             BLOCK_N: tl.constexpr):
    """mqa_attention_aot in one pass (online softmax): unnormalized probabilities round to
    HALF before PV instead of normalized ones. Approximate; for the 521-row prefill."""
    head = tl.program_id(1)
    rows = tl.program_id(0)*BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    keys = tl.arange(0, BLOCK_N)
    valid = rows < N_Q
    q = tl.load(Q + head*N_Q*D + rows[:, None]*D + dims[None, :], mask=valid[:, None], other=0)
    maximum = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accum = tl.zeros((BLOCK_M, D), tl.float32)
    for start in range(0, N_KV, BLOCK_N):
        cols = start + keys
        inside = cols < N_KV
        kt = tl.load(K + cols[None, :]*D + dims[:, None], mask=inside[None, :], other=0)
        s = (tl.dot(q, kt).to(tl.float16).to(tl.float32)*SCALE).to(tl.float16).to(tl.float32)
        logit = s + tl.load(Mask + rows[:, None]*N_KV + cols[None, :], mask=valid[:, None] & inside[None, :], other=0)
        logit = tl.where(inside[None, :], logit, float('-inf'))
        new_max = tl.maximum(maximum, tl.max(logit, 1))
        w = tl.exp(logit - new_max[:, None])
        alpha = tl.exp(maximum - new_max)
        denominator = denominator*alpha + tl.sum(w, 1)
        v = tl.load(V + cols[:, None]*D + dims[None, :], mask=inside[:, None], other=0)
        accum = tl.dot(w.to(tl.float16), v, accum*alpha[:, None])
        maximum = new_max
    tl.store(O + rows[:, None]*(H*D) + head*D + dims[None, :], (accum/denominator[:, None]).to(tl.float16), mask=valid[:, None])
