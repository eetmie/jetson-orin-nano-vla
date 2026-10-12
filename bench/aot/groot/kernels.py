"""GR00T attention on native projection layouts, HALF boundaries and FP32 accumulation."""
import triton
import triton.language as tl


@triton.jit
def _scores(q, k, QK_SCALE: tl.constexpr, SCORE_SCALE: tl.constexpr):
    q = (q.to(tl.float32) * QK_SCALE).to(tl.float16)
    k = (k.to(tl.float32) * QK_SCALE).to(tl.float16)
    return (tl.dot(q, k).to(tl.float16).to(tl.float32) * SCORE_SCALE).to(tl.float16).to(tl.float32)


@triton.jit
def attention_aot(Q, K, V, Bias, O, B: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
                  H: tl.constexpr, D: tl.constexpr, KD: tl.constexpr,
                  QK_SCALE: tl.constexpr, SCORE_SCALE: tl.constexpr,
                  PACKED_KV: tl.constexpr, HAS_BIAS: tl.constexpr, TWO_PASS: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, V_PACKED: tl.constexpr = False,
                  SCORE_FP32: tl.constexpr = False):
    """Q [B,NQ,H*D], K/V [B,NK,H*D] or K|V [B,NK,2*H*D], Bias [1,1,NK].

    SCORE_FP32 computes Q K^T * QK_SCALE^2 * SCORE_SCALE in FP32 instead of the graph's
    FP16 pre-scaling and score roundings (see causal_aot).

    TWO_PASS rounds normalized probabilities to HALF exactly as the exported graph;
    online mode rounds unnormalized weights before PV and needs full-policy parity.
    """
    head, batch = tl.program_id(1), tl.program_id(2)
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    dims = tl.arange(0, KD)
    keys = tl.arange(0, BN)
    width: tl.constexpr = H*D
    stride: tl.constexpr = width*(2 if PACKED_KV else 1)
    qb = Q + batch*NQ*width + head*D
    kb = K + batch*NK*stride + head*D
    vstride: tl.constexpr = width*(3 if V_PACKED else 2 if PACKED_KV else 1)
    vb = (K + width if PACKED_KV else V + 2*width if V_PACKED else V) + batch*NK*vstride + head*D
    q = tl.load(qb + rows[:, None]*width + dims[None, :],
                mask=(rows[:, None]<NQ)&(dims[None, :]<D), other=0)
    maximum = tl.full((BM,), float('-inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accum = tl.zeros((BM, KD), tl.float32)
    for start in range(0, NK, BN):
        cols = start + keys
        k = tl.load(kb + cols[None, :]*stride + dims[:, None],
                    mask=(cols[None, :]<NK)&(dims[:, None]<D), other=0).to(tl.float16)
        score = tl.dot(q, k)*(QK_SCALE*QK_SCALE*SCORE_SCALE) if SCORE_FP32 else _scores(q, k, QK_SCALE, SCORE_SCALE)
        if HAS_BIAS:
            bias = tl.load(Bias + cols, mask=cols<NK, other=0).to(tl.float16).to(tl.float32)
            score = (score+bias[None, :]).to(tl.float16).to(tl.float32)
        score = tl.where(cols[None, :]<NK, score, float('-inf'))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        weight = tl.exp(score-next_max[:, None])
        alpha = tl.exp(maximum-next_max)
        denominator = denominator*alpha + tl.sum(weight, 1)
        if not TWO_PASS:
            value = tl.load(vb + cols[:, None]*vstride + dims[None, :],
                            mask=(cols[:, None]<NK)&(dims[None, :]<D), other=0).to(tl.float16)
            accum = tl.dot(weight.to(tl.float16), value, accum*alpha[:, None])
        maximum = next_max
    if TWO_PASS:
        for start in range(0, NK, BN):
            cols = start + keys
            k = tl.load(kb + cols[None, :]*stride + dims[:, None],
                        mask=(cols[None, :]<NK)&(dims[:, None]<D), other=0).to(tl.float16)
            score = tl.dot(q, k)*(QK_SCALE*QK_SCALE*SCORE_SCALE) if SCORE_FP32 else _scores(q, k, QK_SCALE, SCORE_SCALE)
            if HAS_BIAS:
                bias = tl.load(Bias + cols, mask=cols<NK, other=0).to(tl.float16).to(tl.float32)
                score = (score+bias[None, :]).to(tl.float16).to(tl.float32)
            score = tl.where(cols[None, :]<NK, score, float('-inf'))
            weight = (tl.exp(score-maximum[:, None])/denominator[:, None]).to(tl.float16)
            value = tl.load(vb + cols[:, None]*vstride + dims[None, :],
                            mask=(cols[:, None]<NK)&(dims[None, :]<D), other=0).to(tl.float16)
            accum = tl.dot(weight, value, accum)
        result = accum
    else:
        result = accum/denominator[:, None]
    tl.store(O+batch*NQ*width+rows[:, None]*width+head*D+dims[None, :], result,
             mask=(rows[:, None]<NQ)&(dims[None, :]<D))


@triton.jit
def _scores72(q, qt, k, kt, SCALE: tl.constexpr, SCORE_FP32: tl.constexpr = False):
    dot = tl.dot(q, k)
    dot = tl.dot(qt, kt, dot)
    if SCORE_FP32:
        return dot*SCALE
    return (dot.to(tl.float16).to(tl.float32)*SCALE).to(tl.float16).to(tl.float32)


@triton.jit
def attention72_aot(Q, K, V, Bias, O, B: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
                    H: tl.constexpr, D: tl.constexpr, KD: tl.constexpr,
                    QK_SCALE: tl.constexpr, SCORE_SCALE: tl.constexpr,
                    PACKED_KV: tl.constexpr, HAS_BIAS: tl.constexpr, TWO_PASS: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, V_PACKED: tl.constexpr = False,
                    SCORE_FP32: tl.constexpr = False):
    """SigLIP2 D=72 as a 64-wide tensor-core dot plus a masked 16-wide tail.

    Avoids padding all QK/PV dot products to 128 channels. HALF score boundaries
    and FP32 accumulation are retained; the padded tail consists only of zeros.
    """
    tl.static_assert(D==72 and not PACKED_KV and not HAS_BIAS and not V_PACKED and QK_SCALE==1.)
    head,batch=tl.program_id(1),tl.program_id(2)
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    dims=tl.arange(0,64);tail=64+tl.arange(0,16);keys=tl.arange(0,BN)
    width:tl.constexpr=H*D
    qb=Q+batch*NQ*width+head*D;kb=K+batch*NK*width+head*D;vb=V+batch*NK*width+head*D
    q=tl.load(qb+rows[:,None]*width+dims[None,:],mask=rows[:,None]<NQ,other=0)
    qt=tl.load(qb+rows[:,None]*width+tail[None,:],mask=(rows[:,None]<NQ)&(tail[None,:]<D),other=0)
    maximum=tl.full((BM,),float('-inf'),tl.float32);denominator=tl.zeros((BM,),tl.float32)
    accum=tl.zeros((BM,64),tl.float32);accumt=tl.zeros((BM,16),tl.float32)
    for start in range(0,NK,BN):
        cols=start+keys
        k=tl.load(kb+cols[None,:]*width+dims[:,None],mask=cols[None,:]<NK,other=0)
        kt=tl.load(kb+cols[None,:]*width+tail[:,None],mask=(cols[None,:]<NK)&(tail[:,None]<D),other=0)
        score=_scores72(q,qt,k,kt,SCORE_SCALE,SCORE_FP32)
        score=tl.where(cols[None,:]<NK,score,float('-inf'))
        next_max=tl.maximum(maximum,tl.max(score,1));weight=tl.exp(score-next_max[:,None])
        alpha=tl.exp(maximum-next_max);denominator=denominator*alpha+tl.sum(weight,1)
        if not TWO_PASS:
            v=tl.load(vb+cols[:,None]*width+dims[None,:],mask=cols[:,None]<NK,other=0)
            vt=tl.load(vb+cols[:,None]*width+tail[None,:],mask=(cols[:,None]<NK)&(tail[None,:]<D),other=0)
            p=weight.to(tl.float16)
            accum=tl.dot(p,v,accum*alpha[:,None]);accumt=tl.dot(p,vt,accumt*alpha[:,None])
        maximum=next_max
    if TWO_PASS:
        for start in range(0,NK,BN):
            cols=start+keys
            k=tl.load(kb+cols[None,:]*width+dims[:,None],mask=cols[None,:]<NK,other=0)
            kt=tl.load(kb+cols[None,:]*width+tail[:,None],mask=(cols[None,:]<NK)&(tail[:,None]<D),other=0)
            score=_scores72(q,qt,k,kt,SCORE_SCALE,SCORE_FP32)
            score=tl.where(cols[None,:]<NK,score,float('-inf'))
            p=(tl.exp(score-maximum[:,None])/denominator[:,None]).to(tl.float16)
            v=tl.load(vb+cols[:,None]*width+dims[None,:],mask=cols[:,None]<NK,other=0)
            vt=tl.load(vb+cols[:,None]*width+tail[None,:],mask=(cols[:,None]<NK)&(tail[None,:]<D),other=0)
            accum=tl.dot(p,v,accum);accumt=tl.dot(p,vt,accumt)
        result=accum;resultt=accumt
    else:result=accum/denominator[:,None];resultt=accumt/denominator[:,None]
    base=O+batch*NQ*width+rows[:,None]*width+head*D
    tl.store(base+dims[None,:],result,mask=rows[:,None]<NQ)
    tl.store(base+tail[None,:],resultt,mask=(rows[:,None]<NQ)&(tail[None,:]<D))


@triton.jit
def causal_aot(Q, K, V, Bias, O, S: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
               D: tl.constexpr, SCALE: tl.constexpr,
               SQH: tl.constexpr, SQS: tl.constexpr, SKH: tl.constexpr, SKS: tl.constexpr,
               SVH: tl.constexpr, SVS: tl.constexpr,
               TWO_PASS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               SCORE_FP32: tl.constexpr = True):
    """Causal GQA attention of the GR00T LLM chunks; O [1,S,H*D].

    The graph adds a constant triu(-1e4) mask to fp16(fp16(Q K^T) * SCALE) and runs an FP32
    softmax: masked weights are exactly 0, so key tiles above the diagonal are skipped and
    the diagonal tile masks with -inf. Q/K/V are read in their producers' layouts (strides
    per head and per row; query head h uses key/value head h // G). TWO_PASS rounds
    normalized probabilities to FP16 before PV, as the graph does. SCORE_FP32 keeps
    Q K^T * SCALE in FP32 instead of the graph's two FP16 roundings: TensorRT's own fused
    MHA does the same, and the graph order measured +5..17 % policy error against stock.
    """
    head = tl.program_id(1)
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    keys = tl.arange(0, BN)
    q = tl.load(Q + head*SQH + rows[:, None]*SQS + dims[None, :], mask=rows[:, None] < S, other=0)
    kb = K + (head // G)*SKH
    vb = V + (head // G)*SVH
    maximum = tl.full((BM,), float('-inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accum = tl.zeros((BM, D), tl.float32)
    end = tl.minimum((tl.program_id(0) + 1)*BM, S)
    for start in range(0, end, BN):
        cols = start + keys
        k = tl.load(kb + cols[None, :]*SKS + dims[:, None], mask=cols[None, :] < S, other=0)
        if SCORE_FP32:
            score = tl.dot(q, k)*SCALE
        else:
            score = (tl.dot(q, k).to(tl.float16).to(tl.float32)*SCALE).to(tl.float16).to(tl.float32)
        score = tl.where((cols[None, :] <= rows[:, None]) & (cols[None, :] < S), score, float('-inf'))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        weight = tl.exp(score - next_max[:, None])
        alpha = tl.exp(maximum - next_max)
        denominator = denominator*alpha + tl.sum(weight, 1)
        if not TWO_PASS:
            v = tl.load(vb + cols[:, None]*SVS + dims[None, :], mask=cols[:, None] < S, other=0)
            accum = tl.dot(weight.to(tl.float16), v, accum*alpha[:, None])
        maximum = next_max
    if TWO_PASS:
        for start in range(0, end, BN):
            cols = start + keys
            k = tl.load(kb + cols[None, :]*SKS + dims[:, None], mask=cols[None, :] < S, other=0)
            if SCORE_FP32:
                score = tl.dot(q, k)*SCALE
            else:
                score = (tl.dot(q, k).to(tl.float16).to(tl.float32)*SCALE).to(tl.float16).to(tl.float32)
            score = tl.where((cols[None, :] <= rows[:, None]) & (cols[None, :] < S), score, float('-inf'))
            weight = (tl.exp(score - maximum[:, None])/denominator[:, None]).to(tl.float16)
            v = tl.load(vb + cols[:, None]*SVS + dims[None, :], mask=cols[:, None] < S, other=0)
            accum = tl.dot(weight, v, accum)
        result = accum
    else:
        result = accum/denominator[:, None]
    tl.store(O + rows[:, None]*(H*D) + head*D + dims[None, :], result.to(tl.float16),
             mask=rows[:, None] < S)
