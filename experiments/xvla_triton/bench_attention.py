"""X-VLA denoise attention: Triton xvla_attention_aot vs a torch oracle of the exported graph
(fp16 QK, fp32 softmax, fp16 probs, fp16 PV) and vs TensorRT's own fused MHA engine."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernels import xvla_attention_aot
S, H, D, SC = 262, 16, 64, 0.353515625
torch.manual_seed(2)
qkv = (torch.randn(1, S, 3*H*D, device='cuda')*1.2).half()
def oracle(nq):
    x = qkv.view(1, S, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = x[0]*torch.tensor(SC).half().cuda(), x[1]*torch.tensor(SC).half().cuda(), x[2]
    sc = (q.float() @ k.float().transpose(-1, -2)).half()
    p = torch.softmax(sc.float(), -1).half()
    o = (p.float() @ v.float()).half()
    return o.permute(0, 2, 1, 3).reshape(1, S, H*D)[:, :nq]
def run(nq, bm, bn, w, st, out):
    xvla_attention_aot[(triton_cdiv(nq, bm), H)](qkv, out, N_Q=nq, N_KV=S, H=H, SCALE=SC, BLOCK_M=bm,
                                                 BLOCK_N=bn, D=D, num_warps=w, num_stages=st)
def triton_cdiv(a, b): return (a+b-1)//b
def bench(fn):
    fn(); torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(100): fn()
    for _ in range(3): g.replay()
    ts = []
    for _ in range(9):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b)/100)
    return sorted(ts)[4]
rows = []
for nq in (262, 30):
    ref = oracle(nq)
    for bm, bn, w, st in [(64,64,4,2),(64,64,4,3),(32,64,4,2),(32,64,2,2),(32,64,2,3),(64,32,4,2),(16,64,1,2),(32,32,2,2),(128,64,4,2),(64,128,4,2)]:
        out = torch.empty(1, nq, H*D, device='cuda', dtype=torch.half)
        try:
            t = bench(lambda: run(nq, bm, bn, w, st, out))
        except Exception as e:
            print('skip', nq, bm, bn, w, st, type(e).__name__, str(e)[:60]); continue
        err = (out.float()-ref.float()).abs().max().item()
        rows.append(dict(n_q=nq, bm=bm, bn=bn, w=w, st=st, ms=t, max_abs_vs_oracle=err, finite=bool(torch.isfinite(out).all())))
        print(rows[-1], flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
