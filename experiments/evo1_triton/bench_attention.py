"""EVO1 vision attention: Triton kernel vs a torch oracle of the exported graph."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernels import evo1_vision_attention_aot
B, S, H, D = 2, 1025, 16, 64
SC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.125
torch.manual_seed(4)
q, k, v = [(torch.randn(B, S, H*D, device='cuda')*1.5).half() for _ in range(3)]
def oracle():
    t = lambda x: x.view(B, S, H, D).permute(0, 2, 1, 3)
    sc = ((t(q).float() @ t(k).float().transpose(-1, -2)).half()*torch.tensor(SC).half().cuda())
    p = torch.softmax(sc.float(), -1).half()
    return (p.float() @ t(v).float()).half().permute(0, 2, 1, 3).reshape(B, S, H*D)
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
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b)/20)
    return sorted(ts)[3]
ref = oracle(); rows = []
for bm, bn, w, st in [(64,64,4,2),(64,64,4,3),(128,64,4,2),(128,64,4,3),(128,64,8,3),(64,128,4,2),(128,128,8,2),(64,32,4,3)]:
    o = torch.empty_like(q)
    fn = lambda: evo1_vision_attention_aot[((S+bm-1)//bm, H, B)](q, k, v, o, S=S, H=H, SCALE=SC, BLOCK_M=bm, BLOCK_N=bn, D=D, num_warps=w, num_stages=st)
    try:
        t = bench(fn)
    except Exception as e:
        print('skip', bm, bn, w, st, type(e).__name__); continue
    rows.append(dict(bm=bm, bn=bn, w=w, st=st, ms=t, max_abs_vs_oracle=(o.float()-ref.float()).abs().max().item()))
    print(rows[-1], flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
