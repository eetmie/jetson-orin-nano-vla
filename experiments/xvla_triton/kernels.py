"""X-VLA denoiser attention for SM87, on the fused QKV projection.

The exported block computes QKV = x @ W + b ([1,S,3*H*D], columns Q|K|V, head-major),
scales Q and K by HALF 0.3535 each, takes HALF QK^T, an FP32 softmax rounded to HALF,
and HALF P @ V, with no mask and no NaN guard. Here: the same HALF scaling and HALF
score rounding, FP32 online softmax and accumulation; unnormalized probabilities round
to HALF before PV (approximate; full-action gates required). S need not be a tile
multiple. N_Q < N_KV computes only the first N_Q query rows.
"""
import triton
import triton.language as tl


@triton.jit
def xvla_attention_aot(QKV, O, N_Q: tl.constexpr, N_KV: tl.constexpr, H: tl.constexpr,
                       SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       D: tl.constexpr):
    stride = 3*H*D
    head = tl.program_id(1)
    rows = tl.program_id(0)*BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    keys = tl.arange(0, BLOCK_N)
    q_base, k_base, v_base = head*D, H*D + head*D, 2*H*D + head*D
    valid = rows < N_Q
    q = (tl.load(QKV+q_base+rows[:, None]*stride+dims[None, :], mask=valid[:, None], other=0)*SCALE).to(tl.float16)
    maximum = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accum = tl.zeros((BLOCK_M, D), tl.float32)
    for start in range(0, N_KV, BLOCK_N):
        cols = start+keys
        inside = cols < N_KV
        kt = (tl.load(QKV+k_base+cols[None, :]*stride+dims[:, None], mask=inside[None, :], other=0)*SCALE).to(tl.float16)
        score = tl.dot(q, kt).to(tl.float16).to(tl.float32)
        if N_KV % BLOCK_N != 0:
            score = tl.where(inside[None, :], score, float('-inf'))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        weight = tl.exp(score-next_max[:, None])
        alpha = tl.exp(maximum-next_max)
        denominator = denominator*alpha+tl.sum(weight, 1)
        accum = accum*alpha[:, None]
        value = tl.load(QKV+v_base+cols[:, None]*stride+dims[None, :], mask=inside[:, None], other=0)
        accum = tl.dot(weight.to(tl.float16), value, accum)
        maximum = next_max
    out = accum/denominator[:, None]
    tl.store(O+head*D+rows[:, None]*(H*D)+dims[None, :], out, mask=valid[:, None])


@triton.jit
def xvla_attention_rows_aot(QKV, ROWS, O, N_Q: tl.constexpr, N_KV: tl.constexpr, H: tl.constexpr,
                            SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            D: tl.constexpr):
    """xvla_attention_aot for the first N_Q rows; ROWS only carries the output shape."""
    xvla_attention_aot(QKV, O, N_Q, N_KV, H, SCALE, BLOCK_M, BLOCK_N, D)


@triton.jit
def dwconv_tokens_aot(X, W, B, Y, N_PIX: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
                      BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    """y = x + HALF(dwconv3x3(x) + b), in the token layout [views, H*W, C] (channels last).

    DaViT's positional conv is a 3x3 depthwise convolution (stride 1, zero padding) whose
    output is rounded to HALF and added to its input in HALF. The export runs it in NCHW
    between two transposes; here the taps accumulate in FP32 and nothing is transposed.
    """
    view = tl.program_id(2)
    pix = tl.program_id(0)*BLOCK_P + tl.arange(0, BLOCK_P)
    ch = tl.program_id(1)*BLOCK_C + tl.arange(0, BLOCK_C)
    row, col = pix // WIDTH, pix % WIDTH
    height = N_PIX // WIDTH
    base = X + view*N_PIX*C
    inside = pix < N_PIX
    acc = tl.zeros((BLOCK_P, BLOCK_C), tl.float32)
    for dy in tl.static_range(3):
        for dx in tl.static_range(3):
            r, c = row + dy - 1, col + dx - 1
            ok = inside & (r >= 0) & (r < height) & (c >= 0) & (c < WIDTH)
            x = tl.load(base + (r*WIDTH + c)[:, None]*C + ch[None, :], mask=ok[:, None], other=0)
            acc += x.to(tl.float32)*tl.load(W + ch*9 + dy*3 + dx).to(tl.float32)[None, :]
    conv = (acc + tl.load(B + ch).to(tl.float32)[None, :]).to(tl.float16)
    res = tl.load(base + pix[:, None]*C + ch[None, :], mask=inside[:, None], other=0)
    out = (conv.to(tl.float32) + res.to(tl.float32)).to(tl.float16)
    tl.store(Y + view*N_PIX*C + pix[:, None]*C + ch[None, :], out, mask=inside[:, None])
