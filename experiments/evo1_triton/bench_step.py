"""Sweep the EVO1 action-step cross-attention kernel against a Torch oracle (and SDPA)."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import triton
from kernels import step_attention_aot

p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); a = p.parse_args()
if a.out.exists(): raise FileExistsError(a.out)
torch.manual_seed(20261012)
NQ, NK, H, D = 50, 577, 8, 112
scale = float(np.float16(D ** -0.5))


def bench(fn):
    fn(); torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(20): fn()
    for _ in range(3): g.replay()
    t = []
    for _ in range(7):
        x, y = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        x.record(); g.replay(); y.record(); y.synchronize(); t.append(x.elapsed_time(y) / 20)
    return float(np.median(t))


q = torch.randn(1, NQ, H * D, device='cuda').half()              # q-proj layout
k = torch.randn(1, H, NK, D, device='cuda').half()
v = torch.randn(1, H, NK, D, device='cuda').half()
mask = torch.zeros(NK, device='cuda').half(); mask[500:] = -10000
qq = q.reshape(1, NQ, H, D).transpose(1, 2)


def oracle(fp32):
    sc = qq.float() @ k.float().transpose(-1, -2)
    if fp32:
        sc = sc * scale + mask.float()
    else:
        sc = ((sc.half().float() * scale).half().float() + mask.float()).half().float()
    pr = torch.softmax(sc, -1).half()
    return (pr.float() @ v.float()).half().transpose(1, 2).reshape(1, NQ, H * D)


def sdpa():
    return torch.nn.functional.scaled_dot_product_attention(qq, k, v, attn_mask=mask.view(1, 1, 1, NK))


rows = [dict(kernel='torch_sdpa', median_ms=bench(sdpa))]
o = torch.empty_like(q)
for f32 in (True, False):
    ref = oracle(f32)
    for two in (False, True):
        for bm, bn, w, st in [(64, 32, 4, 1), (64, 32, 4, 2), (64, 64, 4, 1), (32, 64, 4, 1), (32, 32, 4, 2), (64, 16, 4, 2), (64, 32, 8, 2)]:
            try:
                def fn():
                    return step_attention_aot[((NQ + bm - 1) // bm, H)](q, k, v, mask, o, NQ=NQ, NK=NK, H=H, D=D, KD=128, SCALE=scale,
                        SQH=D, SQS=H * D, SKH=NK * D, SKS=D, SVH=NK * D, SVS=D, SCORE_FP32=f32, TWO_PASS=two, BM=bm, BN=bn,
                        num_warps=w, num_stages=st)
                kern = fn(); torch.cuda.synchronize()
                d = (o.float() - ref.float()).abs()
                row = dict(score_fp32=f32, two_pass=two, bm=bm, bn=bn, warps=w, stages=st, shared=kern.metadata.shared,
                           max_abs=float(d.max()), exact=float((d == 0).float().mean()))
                row['status'] = 'rejected_shared' if kern.metadata.shared > 49152 else 'pass'
                if row['status'] == 'pass': row['median_ms'] = bench(fn)
            except Exception as e:
                row = dict(score_fp32=f32, two_pass=two, bm=bm, bn=bn, status='failed', error=str(e)[:200])
            rows.append(row); print(json.dumps(row), flush=True)
a.out.write_text(json.dumps(dict(triton=triton.__version__, rows=rows), indent=2) + '\n')
