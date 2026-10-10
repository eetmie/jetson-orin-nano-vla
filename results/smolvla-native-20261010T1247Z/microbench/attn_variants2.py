"""Audit microbench: SmolVLA vision attention variants on SM87, real shape [1,12,1024,64].

A  = production aligned kernel (attention_kernel.attention_aot), zero mask.
B  = A without the mask load (mask is a constant all-true in the exported graph).
C  = B with n_ctx constexpr (no bounds masks).
D  = C reading Q/K/V and writing O in the projection's native [1,S,H,D] layout,
     applying the 0.3535 FP16 scale to Q and K in-kernel (exported graph scales both).
Bit-exactness of B/C/D is checked against A fed with the same tensors.
"""
import json, sys
from pathlib import Path
import torch, triton, triton.language as tl

sys.path.insert(0, str(Path.home()/'jetson-orin-nano-vla/experiments/smolvla_triton'))
from attention_kernel import attention_aot


@triton.jit
def attn_nomask(Q, Kt, V, n_ctx, O, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    head = tl.program_id(1)
    rows = tl.program_id(0)*BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    keys = tl.arange(0, BLOCK_N)
    base = head*n_ctx*D
    q = tl.load(Q+base+rows[:, None]*D+dims[None, :], mask=rows[:, None] < n_ctx, other=0)
    maximum = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accum = tl.zeros((BLOCK_M, D), tl.float32)
    bad = tl.full((BLOCK_M,), False, tl.int1)
    for start in range(0, n_ctx, BLOCK_N):
        cols = start+keys
        kt = tl.load(Kt+base+dims[:, None]*n_ctx+cols[None, :], mask=cols[None, :] < n_ctx, other=0)
        score = tl.dot(q, kt).to(tl.float16).to(tl.float32)
        score = tl.where(cols[None, :] < n_ctx, score, float('-inf'))
        bad = bad | (tl.sum(((score != score) | (score == float('inf'))).to(tl.int32), 1) > 0)
        next_max = tl.maximum(maximum, tl.max(score, 1))
        shift = tl.where(next_max == float('-inf'), 0.0, next_max)
        weight = tl.exp(score-shift[:, None])
        alpha = tl.exp(maximum-shift)
        denominator = denominator*alpha+tl.sum(weight, 1)
        accum = accum*alpha[:, None]
        value = tl.load(V+base+cols[:, None]*D+dims[None, :], mask=cols[:, None] < n_ctx, other=0)
        accum = tl.dot(weight.to(tl.float16), value, accum)
        maximum = next_max
    out = accum/tl.where(denominator > 0, denominator, 1.0)[:, None]
    out = tl.where(bad[:, None], 0.0, out)
    tl.store(O+base+rows[:, None]*D+dims[None, :], out, mask=rows[:, None] < n_ctx)


@triton.jit
def attn_static(Q, Kt, V, O, N_CTX: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                D: tl.constexpr):
    head = tl.program_id(1)
    rows = tl.program_id(0)*BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    keys = tl.arange(0, BLOCK_N)
    base = head*N_CTX*D
    q = tl.load(Q+base+rows[:, None]*D+dims[None, :])
    maximum = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accum = tl.zeros((BLOCK_M, D), tl.float32)
    bad = tl.full((BLOCK_M,), False, tl.int1)
    for start in range(0, N_CTX, BLOCK_N):
        cols = start+keys
        kt = tl.load(Kt+base+dims[:, None]*N_CTX+cols[None, :])
        score = tl.dot(q, kt).to(tl.float16).to(tl.float32)
        bad = bad | (tl.sum(((score != score) | (score == float('inf'))).to(tl.int32), 1) > 0)
        next_max = tl.maximum(maximum, tl.max(score, 1))
        shift = tl.where(next_max == float('-inf'), 0.0, next_max)
        weight = tl.exp(score-shift[:, None])
        alpha = tl.exp(maximum-shift)
        denominator = denominator*alpha+tl.sum(weight, 1)
        accum = accum*alpha[:, None]
        value = tl.load(V+base+cols[:, None]*D+dims[None, :])
        accum = tl.dot(weight.to(tl.float16), value, accum)
        maximum = next_max
    out = accum/tl.where(denominator > 0, denominator, 1.0)[:, None]
    out = tl.where(bad[:, None], 0.0, out)
    tl.store(O+base+rows[:, None]*D+dims[None, :], out)


@triton.jit
def attn_native(Q, K, V, O, scale, N_CTX: tl.constexpr, H: tl.constexpr, BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr, D: tl.constexpr):
    # Q/K/V/O are [1, S, H, D] contiguous (row stride H*D); scale is the graph's FP16 0.3535.
    head = tl.program_id(1)
    rows = tl.program_id(0)*BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    keys = tl.arange(0, BLOCK_N)
    stride = H*D
    base = head*D
    s = scale.to(tl.float16)
    q = (tl.load(Q+base+rows[:, None]*stride+dims[None, :])*s).to(tl.float16)
    maximum = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accum = tl.zeros((BLOCK_M, D), tl.float32)
    bad = tl.full((BLOCK_M,), False, tl.int1)
    for start in range(0, N_CTX, BLOCK_N):
        cols = start+keys
        kt = (tl.load(K+base+cols[None, :]*stride+dims[:, None])*s).to(tl.float16)
        score = tl.dot(q, kt).to(tl.float16).to(tl.float32)
        bad = bad | (tl.sum(((score != score) | (score == float('inf'))).to(tl.int32), 1) > 0)
        next_max = tl.maximum(maximum, tl.max(score, 1))
        shift = tl.where(next_max == float('-inf'), 0.0, next_max)
        weight = tl.exp(score-shift[:, None])
        alpha = tl.exp(maximum-shift)
        denominator = denominator*alpha+tl.sum(weight, 1)
        accum = accum*alpha[:, None]
        value = tl.load(V+base+cols[:, None]*stride+dims[None, :])
        accum = tl.dot(weight.to(tl.float16), value, accum)
        maximum = next_max
    out = accum/tl.where(denominator > 0, denominator, 1.0)[:, None]
    out = tl.where(bad[:, None], 0.0, out)
    tl.store(O+base+rows[:, None]*stride+dims[None, :], out)

@triton.jit
def attn_native2(Q, K, V, O, scale, N_CTX: tl.constexpr, H: tl.constexpr, BLOCK_M: tl.constexpr,
                 BLOCK_N: tl.constexpr, D: tl.constexpr):
    head = tl.program_id(1)
    rows = tl.program_id(0)*BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    keys = tl.arange(0, BLOCK_N)
    stride = H*D
    base = head*D
    s = scale.to(tl.float16)
    q = (tl.load(Q+base+rows[:, None]*stride+dims[None, :])*s).to(tl.float16)
    maximum = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accum = tl.zeros((BLOCK_M, D), tl.float32)
    bad = tl.full((BLOCK_M,), False, tl.int1)
    for start in range(0, N_CTX, BLOCK_N):
        cols = start+keys
        k = (tl.load(K+base+cols[:, None]*stride+dims[None, :])*s).to(tl.float16)
        score = tl.dot(q, tl.trans(k)).to(tl.float16).to(tl.float32)
        bad = bad | (tl.sum(((score != score) | (score == float('inf'))).to(tl.int32), 1) > 0)
        next_max = tl.maximum(maximum, tl.max(score, 1))
        shift = tl.where(next_max == float('-inf'), 0.0, next_max)
        weight = tl.exp(score-shift[:, None])
        alpha = tl.exp(maximum-shift)
        denominator = denominator*alpha+tl.sum(weight, 1)
        accum = accum*alpha[:, None]
        value = tl.load(V+base+cols[:, None]*stride+dims[None, :])
        accum = tl.dot(weight.to(tl.float16), value, accum)
        maximum = next_max
    out = accum/tl.where(denominator > 0, denominator, 1.0)[:, None]
    out = tl.where(bad[:, None], 0.0, out)
    tl.store(O+base+rows[:, None]*stride+dims[None, :], out)


def bench(fn, stream, reps=50, batches=9):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        for _ in range(reps):
            fn()
    times = []
    for _ in range(batches):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b)/reps)
    times.sort()
    return times[len(times)//2]


NAMES = ['C static', 'D native', 'D2 native-trans']

def main():
    out_path = Path(sys.argv[1])
    assert torch.cuda.get_device_capability() == (8, 7)
    torch.manual_seed(20261010)
    S, H, D = 1024, 12, 64
    scale = torch.tensor(0.3535, dtype=torch.float16)
    # raw projection outputs in native layout, magnitudes like the real QKV output
    qn = torch.randn(1, S, H, D, device='cuda', dtype=torch.float16)*1.5
    kn = torch.randn(1, S, H, D, device='cuda', dtype=torch.float16)*1.5
    vn = torch.randn(1, S, H, D, device='cuda', dtype=torch.float16)
    # what the production graph feeds the plugin: scaled Q[B,H,S,D], scaled Kt[B,H,D,S], V[B,H,S,D]
    q = (qn*scale).permute(0, 2, 1, 3).contiguous()
    kt = (kn*scale).permute(0, 2, 3, 1).contiguous()
    v = vn.permute(0, 2, 1, 3).contiguous()
    mask = torch.zeros(1, 1, S, S, device='cuda', dtype=torch.float16)
    ref = torch.empty_like(q)
    stream = torch.cuda.Stream()
    res = dict(shape=[1, H, S, D], torch=torch.__version__, triton=triton.__version__, rows=[])
    with torch.cuda.stream(stream):
        A = lambda: attention_aot[(S//128, H)](q, kt, v, mask, S, ref, BLOCK_M=128, BLOCK_N=64, D=D,
                                              num_warps=4, num_stages=2)
        A(); torch.cuda.synchronize()
        t = bench(A, stream)
        res['rows'].append(dict(variant='A production', bm=128, bn=64, w=4, st=2, ms=t, exact=True))
        print(res['rows'][-1], flush=True)
        # transposes production pays per layer (TranMul x2 + Tran on input, Tran on output)
        tq = torch.empty_like(q); tk = torch.empty_like(kt); tv = torch.empty_like(v)
        on = torch.empty_like(qn)
        def T():
            tq.copy_((qn*scale.item()).permute(0, 2, 1, 3)); tk.copy_((kn*scale.item()).permute(0, 2, 3, 1))
            tv.copy_(vn.permute(0, 2, 1, 3)); on.copy_(ref.permute(0, 2, 1, 3))
        res['torch_transposes_ms'] = bench(T, stream)
        print('torch transposes (upper-bound ref):', res['torch_transposes_ms'], flush=True)
        configs = [(128, 64, 4, 2), (128, 64, 4, 3), (64, 64, 4, 2), (64, 64, 4, 3), (64, 64, 2, 2), (64, 64, 2, 3), (32, 64, 2, 3), (64, 128, 4, 2), (64, 128, 4, 3)]
        for name in NAMES:
            for bm, bn, w, st in configs:
                o = torch.empty_like(q) if not name.startswith('D') else torch.empty_like(qn)
                if name == 'B nomask':
                    fn = lambda: attn_nomask[(S//bm, H)](q, kt, v, S, o, BLOCK_M=bm, BLOCK_N=bn, D=D,
                                                         num_warps=w, num_stages=st)
                elif name == 'C static':
                    fn = lambda: attn_static[(S//bm, H)](q, kt, v, o, N_CTX=S, BLOCK_M=bm, BLOCK_N=bn, D=D,
                                                         num_warps=w, num_stages=st)
                elif name == 'D2 native-trans':
                    fn = lambda: attn_native2[(S//bm, H)](qn, kn, vn, o, scale.item(), N_CTX=S, H=H, BLOCK_M=bm,
                                                          BLOCK_N=bn, D=D, num_warps=w, num_stages=st)
                else:
                    fn = lambda: attn_native[(S//bm, H)](qn, kn, vn, o, scale.item(), N_CTX=S, H=H, BLOCK_M=bm,
                                                         BLOCK_N=bn, D=D, num_warps=w, num_stages=st)
                try:
                    fn(); torch.cuda.synchronize()
                except Exception as e:
                    print('skip', name, bm, bn, w, st, type(e).__name__, str(e)[:80]); continue
                got = o if not name.startswith('D') else o.permute(0, 2, 1, 3)
                diff = (got.float()-ref.float()).abs().max().item()
                exact = bool(torch.equal(got, ref)) if bn == 64 else None
                t = bench(fn, stream)
                res['rows'].append(dict(variant=name, bm=bm, bn=bn, w=w, st=st, ms=t, max_abs_vs_A=diff,
                                        bit_exact_vs_A=exact))
                print(res['rows'][-1], flush=True)
    out_path.write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
