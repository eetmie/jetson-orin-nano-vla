"""Sweep both GR00T vision/DiT attention shapes against the export's Torch oracle."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import triton
from kernels import attention_aot

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
p.add_argument('--set',choices=['original','gathered'],default='original',
               help='gathered: cross-attention on K/V gathered to the visible rows (FP16, no mask)')
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
assert torch.cuda.get_device_capability()==(8,7)
torch.manual_seed(20261011)

def bench(fn):
    fn();torch.cuda.synchronize()
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            for _ in range(20):fn()
    for _ in range(3):graph.replay()
    samples=[]
    for _ in range(7):
        x,y=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        x.record();graph.replay();y.record();y.synchronize();samples.append(x.elapsed_time(y)/20)
    return float(np.median(samples))

records=[]
shapes=[('n16-vision',3,324,324,16,72,False),('n17-vision',3,256,256,16,64,False),
        ('n16-cross',1,51,320,32,48,True),('n17-cross',1,41,448,32,48,True),
        ('n16-self',1,51,51,32,48,False),('n17-self',1,41,41,32,48,False)]
if a.set=='gathered':
    shapes=[('n16-cross48',1,51,48,32,48,True),('n16-cross243',1,51,243,32,48,True),
            ('n17-cross28',1,41,28,32,48,True),('n17-cross384',1,41,384,32,48,True)]
gathered=a.set=='gathered'
for name,b,nq,nk,h,d,packed in shapes:
    sdpa='vision' not in name
    q=torch.randn((b,nq,h*d),device='cuda').half()
    k=torch.randn((b,nk,h*d*(2 if packed else 1)),device='cuda').half()
    v=torch.randn((b,nk,h*d),device='cuda').half()
    if packed and not gathered:k=k.float()  # FP32 [K|V] boundary of full-length bundles.
    masked=packed and not gathered
    bias=torch.zeros(nk,device='cuda');bias[nk-25:]=-10000 if masked else 0
    qs=float(np.float16(np.sqrt(np.float16(np.float16(1)/np.float16(np.sqrt(np.float16(d))))))) if sdpa else 1.
    ss=1. if sdpa else float(np.float16(d**-.5))
    def oracle():
        qq=q.reshape(b,nq,h,d).transpose(1,2)
        kk=(k[...,:h*d] if packed else k).half().reshape(b,nk,h,d).transpose(1,2)
        vv=(k[...,h*d:] if packed else v).half().reshape(b,nk,h,d).transpose(1,2)
        qq=(qq.float()*qs).half();kk=(kk.float()*qs).half()
        score=((qq.float()@kk.float().transpose(-1,-2)).half().float()*ss).half()
        if masked:score=(score.float()+bias.half().float()).half()
        prob=torch.softmax(score.float(),-1).half()
        return (prob.float()@vv.float()).half().transpose(1,2).contiguous().reshape(b,nq,h*d)
    ref=oracle();out=torch.empty_like(q)
    rows=[]
    for two in [True,False]:
        for bm,bn,w,st in [(32,32,4,2),(32,64,4,2),(64,32,4,2),(64,64,4,2),(128,32,4,2),(128,64,4,2),(64,64,8,2)]:
            try:
                def fn():
                    return attention_aot[((nq+bm-1)//bm,h,b)](q,k,v,bias,out,B=b,NQ=nq,NK=nk,H=h,D=d,KD=triton.next_power_of_2(d),QK_SCALE=qs,SCORE_SCALE=ss,PACKED_KV=packed,HAS_BIAS=masked,TWO_PASS=two,BM=bm,BN=bn,num_warps=w,num_stages=st)
                kernel=fn();torch.cuda.synchronize()
                shared=kernel.metadata.shared
                delta=(out-ref).float()
                row=dict(two_pass=two,bm=bm,bn=bn,warps=w,stages=st,shared_bytes=shared,max_abs=float(delta.abs().max()),cosine=float(torch.nn.functional.cosine_similarity(out.float().flatten(),ref.float().flatten(),dim=0)))
                if shared>49152:row.update(status='rejected_shared_memory')
                elif not bool(torch.isfinite(out).all()) or row['cosine'] < .99999 or row['max_abs'] > .002:
                    row.update(status='rejected_numerical_error')
                else:row.update(status='pass',median_ms=bench(fn))
            except Exception as e:
                row=dict(two_pass=two,bm=bm,bn=bn,warps=w,stages=st,status='failed',error=str(e)[:400])
            rows.append(row)
            print(name,json.dumps(row),flush=True)
    records.append(dict(name=name,shape=[b,nq,nk,h,d],qk_scale=qs,score_scale=ss,rows=rows))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(torch=torch.__version__,triton=triton.__version__,gpu=torch.cuda.get_device_name(),records=records),indent=2)+'\n')
