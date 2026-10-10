"""Fixed-shape expert gated projection with exported HALF rounding boundaries."""
import triton
import triton.language as tl

@triton.jit
def gated_projection(X, Gate, Up, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.program_id(1)*BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for start in range(0, K, BK):
        k = start + kk
        x = tl.load(X+rows[:, None]*K+k[None, :],
                    mask=(rows[:, None]<M)&(k[None, :]<K), other=0)
        g = tl.load(Gate+k[:, None]*N+cols[None, :],
                    mask=(k[:, None]<K)&(cols[None, :]<N), other=0)
        u = tl.load(Up+k[:, None]*N+cols[None, :],
                    mask=(k[:, None]<K)&(cols[None, :]<N), other=0)
        gate = tl.dot(x, g, gate)
        up = tl.dot(x, u, up)
    g = gate.to(tl.float16).to(tl.float32)
    u = up.to(tl.float16).to(tl.float32)
    sigmoid = (1.0/(1.0+tl.exp(-g))).to(tl.float16).to(tl.float32)
    silu = (g*sigmoid).to(tl.float16).to(tl.float32)
    result = (silu*u).to(tl.float16)
    tl.store(Y+rows[:, None]*N+cols[None, :], result,
             mask=(rows[:, None]<M)&(cols[None, :]<N))

@triton.jit
def packed_gated_projection(X, Weight, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.program_id(1)*(2*BN) + tl.arange(0, 2*BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, 2*BN), tl.float32)
    for start in range(0, K, BK):
        k = start+kk
        x = tl.load(X+rows[:, None]*K+k[None, :],
                    mask=(rows[:, None]<M)&(k[None, :]<K), other=0)
        w = tl.load(Weight+k[:, None]*(2*N)+cols[None, :],
                    mask=(k[:, None]<K)&(cols[None, :]<2*N), other=0)
        acc = tl.dot(x,w,acc)
    gate, up = tl.split(tl.reshape(acc, (BM, BN, 2)))
    gate = gate.to(tl.float16).to(tl.float32)
    up = up.to(tl.float16).to(tl.float32)
    sigmoid = (1.0/(1.0+tl.exp(-gate))).to(tl.float16).to(tl.float32)
    silu = (gate*sigmoid).to(tl.float16).to(tl.float32)
    out_cols = tl.program_id(1)*BN+tl.arange(0,BN)
    tl.store(Y+rows[:, None]*N+out_cols[None, :], (silu*up).to(tl.float16),
             mask=(rows[:, None]<M)&(out_cols[None, :]<N))
