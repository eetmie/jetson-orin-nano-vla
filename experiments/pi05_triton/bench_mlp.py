"""π0.5 language MLP gate/up + gated GELU: fused Triton dual GEMM vs torch (cuBLAS) pieces."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernels import gated_mlp_aot
M, K, N = 521, 2048, 16384
torch.manual_seed(7)
x = (torch.randn(M, K, device='cuda')*0.5).half(); wg = (torch.randn(K, N, device='cuda')*0.02).half(); wu = (torch.randn(K, N, device='cuda')*0.02).half()
def ref():
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    g = (x.float() @ wg.float()).half(); u = (x.float() @ wu.float()).half()
    h = lambda t: t.half()
    t = h(g*g); t = h(g*t); t = h(torch.tensor(0.044708251953125).half().cuda()*t); t = h(g+t)
    t = h(torch.tensor(0.7978515625).half().cuda()*t); t = torch.tanh(t.float()).half(); t = h(1+t); t = h(g*t); t = h(0.5*t)
    return t*u
def bench(fn, reps=5):
    fn(); torch.cuda.synchronize()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            for _ in range(reps): fn()
    for _ in range(2): g.replay()
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b)/reps)
    return sorted(ts)[2]
r = ref()
gbuf = torch.empty(M, N, device='cuda', dtype=torch.half); ubuf = torch.empty_like(gbuf)
t_torch = bench(lambda: (torch.mm(x, wg, out=gbuf), torch.mm(x, wu, out=ubuf)))
print('torch two GEMMs (fp32 acc)', round(t_torch, 3), 'ms  | TensorRT gate+gelu 4.86 + up 3.93 + mul 0.53 = 9.32 ms', flush=True)
rows = []
y = torch.empty(M, N, device='cuda', dtype=torch.half)
for bm, bn, bk, w, st in [(64,128,32,4,2),(128,128,32,8,2),(64,128,32,8,2),(128,64,32,4,2),(64,64,64,4,2),(128,128,16,8,2),(64,256,32,8,2),(128,64,32,8,3),(64,128,64,8,2)]:
    fn = lambda: gated_mlp_aot[((M+bm-1)//bm, N//bn)](x, wg, wu, y, M=M, N=N, K=K, BM=bm, BN=bn, BK=bk, num_warps=w, num_stages=st)
    try:
        c = fn(); t = bench(fn)
    except Exception as e:
        print('skip', bm, bn, bk, w, st, type(e).__name__, str(e)[:60]); continue
    d = (y.float()-r.float()).abs()
    rows.append(dict(bm=bm, bn=bn, bk=bk, w=w, st=st, shared=c.metadata.shared, ms=t, max_abs=d.max().item(),
                     frac_exact=(d == 0).float().mean().item()))
    print(rows[-1], flush=True)
Path(sys.argv[1]).write_text(json.dumps(dict(torch_two_gemms_ms=t_torch, rows=rows), indent=1))
