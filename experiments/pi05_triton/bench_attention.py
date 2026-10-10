"""π0.5 MQA attention (decode 10 rows and prefill 521 rows) vs a torch oracle of the exported graph."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernels import mqa_attention_aot, mqa_attention_online_aot
H, D, SC = 8, 256, 0.0625
def oracle(q, k, v, mask):
    s = (q.float() @ k.float().t()).half()            # [H, Nq, Nkv], FP32 accumulate -> HALF
    s = (s*torch.tensor(SC).half().cuda()).float() + mask[None]
    p = torch.softmax(s, -1).half().float()
    o = (p @ v.float()).half()                         # [H, Nq, D]
    return o.permute(1, 0, 2).reshape(q.shape[1], H*D)
def c_smem(kern, nq, nkv, bm, bn, w, st):
    return 0

def bench(fn):
    fn(); torch.cuda.synchronize()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            for _ in range(20): fn()
    for _ in range(3): g.replay()
    ts = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b)/20)
    return sorted(ts)[3]
torch.manual_seed(6)
rows = []
for name, nq, nkv in ((('prefill', 521, 521),) if len(sys.argv) > 2 else (('decode', 10, 531), ('prefill', 521, 521))):
    q = (torch.randn(H, nq, D, device='cuda')*2).half(); k = (torch.randn(nkv, D, device='cuda')*2).half()
    v = torch.randn(nkv, D, device='cuda').half()
    mask = torch.zeros(nq, nkv, device='cuda'); mask[:, nkv-40:nkv-20] = -2.3819763e38   # padded keys
    if name == 'prefill':
        mask += torch.triu(torch.full((nq, nkv), -2.3819763e38, device='cuda'), 1)*(torch.arange(nq, device='cuda')[:, None] > 400)
    ref = oracle(q, k, v, mask)
    cfgs = [(16, 64, 4, 2), (16, 32, 4, 2), (16, 64, 8, 2), (16, 128, 8, 2)] if name == 'decode' else \
           [(32, 64, 4, 2), (64, 64, 8, 2), (64, 32, 4, 2), (32, 32, 4, 2), (64, 32, 8, 1), (64, 64, 8, 1), (128, 32, 8, 1), (32, 64, 4, 1)]
    for bm, bn, w, st in cfgs:
        o = torch.empty(nq, H*D, device='cuda', dtype=torch.half)
        kern = mqa_attention_online_aot if len(sys.argv) > 2 else mqa_attention_aot
        fn = lambda: kern[((nq+bm-1)//bm, H)](q, k, v, mask, o, N_Q=nq, N_KV=nkv, H=H, D=D, SCALE=SC,
                                                           BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=st)
        try:
            t = bench(fn)
        except Exception as e:
            print('skip', name, bm, bn, w, st, type(e).__name__, str(e)[:80]); continue
        d = (o.float()-ref.float()).abs()
        if c_smem(kern, nq, nkv, bm, bn, w, st) > 48*1024: print('over 48K', bm, bn, w, st)
        rows.append(dict(op=name, bm=bm, bn=bn, w=w, st=st, ms=t, max_abs=d.max().item(), frac_exact=(d == 0).float().mean().item()))
        print(rows[-1], flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
