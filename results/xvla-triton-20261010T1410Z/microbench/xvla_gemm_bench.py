"""X-VLA denoise GEMMs at M=262 on SM87: Triton tiles vs torch (cuBLAS), FP32 accumulation."""
import json, sys
from pathlib import Path
import torch, triton, triton.language as tl

@triton.jit
def gemm(X, W, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr, GELU: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm*BM + tl.arange(0, BM); rn = pn*BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        x = tl.load(X + rm[:, None]*K + (k+rk)[None, :], mask=rm[:, None] < M, other=0)
        w = tl.load(W + (k+rk)[:, None]*N + rn[None, :])
        acc = tl.dot(x, w, acc)
    acc = acc + tl.load(B + rn)[None, :].to(tl.float32)
    if GELU:
        inner = 0.7978845608028654*(acc + 0.044715*acc*acc*acc)
        acc = 0.5*acc*(1.0 + (2.0/(1.0+tl.exp(-2.0*inner)) - 1.0))
    tl.store(Y + rm[:, None]*N + rn[None, :], acc.to(tl.float16), mask=rm[:, None] < M)

def bench(fn, reps=50):
    fn(); torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(reps): fn()
    for _ in range(3): g.replay()
    ts = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b)/reps)
    return sorted(ts)[3]

rows = []
M = 262
for name, N, K, gelu in [('fc1+gelu', 4096, 1024, True), ('qkv', 3072, 1024, False), ('fc2', 1024, 4096, False), ('proj', 1024, 1024, False)]:
    x = (torch.randn(M, K, device='cuda')*0.5).half(); w = (torch.randn(K, N, device='cuda')*0.03).half(); bias = (torch.randn(N, device='cuda')*0.1).half()
    y = torch.empty(M, N, device='cuda', dtype=torch.half)
    ref = torch.nn.functional.gelu(x.float()@w.float()+bias.float(), approximate='tanh') if gelu else x.float()@w.float()+bias.float()
    t_torch = bench(lambda: torch.addmm(bias, x, w, out=y))
    print(name, 'torch addmm', round(t_torch, 4), flush=True)
    best = None
    for BM, BN, BK, wp, st in [(64,128,32,4,3),(64,128,64,4,3),(96,128,32,4,3),(64,256,32,8,3),(128,128,32,4,3),(128,128,32,8,3),(64,64,64,4,4),(32,128,64,4,4),(96,128,64,8,3),(128,64,64,4,3),(48,128,64,4,3),(64,128,32,4,4),(64,128,64,4,4)]:
        try:
            fn = lambda: gemm[(triton.cdiv(M, BM), N//BN)](x, w, bias, y, M=M, N=N, K=K, BM=BM, BN=BN, BK=BK, GELU=gelu, num_warps=wp, num_stages=st)
            t = bench(fn)
        except Exception as e:
            print('skip', name, BM, BN, BK, type(e).__name__); continue
        err = ((y.float()-ref).abs().max()/ref.abs().max()).item()
        rows.append(dict(op=name, BM=BM, BN=BN, BK=BK, warps=wp, stages=st, ms=t, torch_ms=t_torch, rel_err=err))
        if best is None or t < best[0]: best = (t, BM, BN, BK, wp, st)
    print(name, 'best triton', best, flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
