"""Sweep the causal GQA LLM attention against a Torch oracle of the export's precision order."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import triton
from kernels import causal_aot

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
torch.manual_seed(20261012)

def bench(fn):
    fn();torch.cuda.synchronize()
    s=torch.cuda.Stream()
    with torch.cuda.stream(s):
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=s):
            for _ in range(20):fn()
    for _ in range(3):g.replay()
    out=[]
    for _ in range(7):
        x,y=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        x.record();g.replay();y.record();y.synchronize();out.append(x.elapsed_time(y)/20)
    return float(np.median(out))

H,HKV,D=16,8,128;G=H//HKV
scale=float(np.float16(D**-.5));neg=-1e4
records=[]
for S in (320,448):
    # Producer layouts seen in the graphs: Q [1,H,S,D] and K [1,HKV,S,D] after RoPE,
    # V straight from v_proj [1,S,HKV*D].
    q=(torch.randn(1,H,S,D,device='cuda')*2).half();k=(torch.randn(1,HKV,S,D,device='cuda')*2).half()
    v=torch.randn(1,S,HKV*D,device='cuda').half()
    mask=torch.full((S,S),neg,device='cuda').triu(1).half()
    def oracle():
        kk=k.repeat_interleave(G,1);vv=v.reshape(1,S,HKV,D).transpose(1,2).repeat_interleave(G,1)
        sc=((q.float()@kk.float().transpose(-1,-2)).half().float()*scale).half()
        sc=(sc.float()+mask.float()).half()
        p=torch.softmax(sc.float(),-1).half()
        return (p.float()@vv.float()).half().transpose(1,2).reshape(1,S,H*D)
    ref=oracle();o=torch.empty(1,S,H*D,device='cuda',dtype=torch.float16)
    # TensorRT-like full masked attention for a speed reference
    def sdpa():
        kk=k.repeat_interleave(G,1);vv=v.reshape(1,S,HKV,D).transpose(1,2).repeat_interleave(G,1)
        return torch.nn.functional.scaled_dot_product_attention(q,kk,vv,attn_mask=mask)
    ds=(sdpa().transpose(1,2).reshape(1,S,H*D).float()-ref.float()).abs()
    gate=max(.004,1.5*float(ds.max()))   # a fused kernel's own reordering vs this oracle
    rows=[dict(kernel='torch_sdpa_masked',median_ms=bench(sdpa),max_abs=float(ds.max()),
               exact_frac=float((ds==0).float().mean()),gate=gate)]
    for two in (True,False):
        for bm,bn,w,st in [(32,32,4,1),(32,64,4,1),(64,32,4,1),(64,64,4,1),(64,32,4,2),(64,64,8,1),(128,32,8,1),(128,64,8,1)]:
            try:
                def fn():
                    return causal_aot[((S+bm-1)//bm,H)](q,k,v,q,o,S=S,H=H,G=G,D=D,SCALE=scale,
                        SQH=S*D,SQS=D,SKH=S*D,SKS=D,SVH=D,SVS=HKV*D,TWO_PASS=two,BM=bm,BN=bn,num_warps=w,num_stages=st)
                kern=fn();torch.cuda.synchronize()
                d=(o.float()-ref.float())
                row=dict(two_pass=two,bm=bm,bn=bn,warps=w,stages=st,shared=kern.metadata.shared,
                         max_abs=float(d.abs().max()),exact_frac=float((d==0).float().mean()))
                if kern.metadata.shared>49152:row['status']='rejected_shared_memory'
                elif not torch.isfinite(o).all() or row['max_abs']>gate:row['status']='rejected_numerical'
                else:row.update(status='pass',median_ms=bench(fn))
            except Exception as e:
                row=dict(two_pass=two,bm=bm,bn=bn,status='failed',error=str(e)[:300])
            rows.append(row);print(S,json.dumps(row),flush=True)
    records.append(dict(S=S,rows=rows))
a.out.write_text(json.dumps(dict(triton=triton.__version__,records=records),indent=2)+'\n')
