"""EVO1 action_output pool GEMV: FP32-accumulating Triton kernel vs torch FP16/FP32 accumulation."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernels import gemv_bias_aot
N, K = 896, 44800
torch.manual_seed(5)
x = (torch.randn(1, K, device='cuda')).half(); w = (torch.randn(N, K, device='cuda')*0.01).half(); b = torch.randn(N, device='cuda').half()
ref = (x.double() @ w.double().t() + b.double())
def bench(fn):
    fn(); torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(20): fn()
    for _ in range(3): g.replay()
    ts = []
    for _ in range(7):
        a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); e.record(); e.synchronize(); ts.append(a.elapsed_time(e)/20)
    return sorted(ts)[3]
rows = []
y = torch.empty(1, N, device='cuda', dtype=torch.half)
for acc16 in (True, False):
    torch.backends.cuda.matmul.allow_fp16_accumulation = acc16
    t = bench(lambda: torch.addmm(b, x, w.t(), out=y))
    rows.append(dict(impl=f'torch fp16_acc={acc16}', ms=t, GBps=N*K*2/t/1e6, max_rel_err=((y.double()-ref).abs().max()/ref.abs().max()).item()))
    print(rows[-1], flush=True)
for bn, bk, wp in [(8,1024,4),(16,512,4),(16,1024,4),(8,2048,4),(4,2048,4),(16,1024,8),(32,512,8),(8,1024,8),(4,4096,8),(8,256,4),(16,256,4),(4,1024,4),(2,2048,4),(8,512,2)]:
    fn = lambda: gemv_bias_aot[(N//bn,)](x, w, b, y, N=N, K=K, BLOCK_N=bn, BLOCK_K=bk, num_warps=wp)
    try:
        t = bench(fn)
    except Exception as e:
        print('skip', bn, bk, wp, type(e).__name__); continue
    rows.append(dict(impl='triton fp32 acc', bn=bn, bk=bk, warps=wp, ms=t, GBps=N*K*2/t/1e6, max_rel_err=((y.double()-ref).abs().max()/ref.abs().max()).item()))
    print(rows[-1], flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
