"""Tile sweep of attention_native_aot at [1,1024,12,64], bit-exactness vs the shipped tile."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.home()/'jetson-orin-nano-vla/experiments/smolvla_triton'))
from attention_kernel import attention_native_aot

S, H, D, SC = 1024, 12, 64, 0.353515625
torch.manual_seed(1)
q, k, v = [torch.randn(1, S, H, D, device='cuda', dtype=torch.float16)*1.5 for _ in range(3)]
def run(bm, bn, w, st, out):
    attention_native_aot[(S//bm, H)](q, k, v, out, N_CTX=S, H=H, SCALE=SC, BLOCK_M=bm, BLOCK_N=bn, D=D,
                                     num_warps=w, num_stages=st)
def bench(bm, bn, w, st):
    out = torch.empty_like(q); run(bm, bn, w, st, out); torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(50): run(bm, bn, w, st, out)
    ts = []
    for _ in range(9):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b)/50)
    return sorted(ts)[4], out
for _ in range(3): bench(64, 64, 4, 3)
ref_t, ref = bench(64, 64, 4, 3)
rows = [dict(bm=64, bn=64, w=4, st=3, ms=ref_t, exact=True)]
print(rows[-1], flush=True)
for bm, bn, w, st in [(128,64,4,3),(128,64,4,2),(128,64,4,4),(128,64,4,5),(64,64,4,2),(64,64,2,2),(128,64,2,3)]:
    try:
        t, o = bench(bm, bn, w, st)
    except Exception as e:
        print('skip', bm, bn, w, st, type(e).__name__); continue
    rows.append(dict(bm=bm, bn=bn, w=w, st=st, ms=t, exact=bool(torch.equal(o, ref)),
                     max_abs=float((o.float()-ref.float()).abs().max())))
    print(rows[-1], flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
