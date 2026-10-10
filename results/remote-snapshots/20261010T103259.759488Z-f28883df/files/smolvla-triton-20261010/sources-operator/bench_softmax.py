"""Compare lab Triton softmax to TRT at SmolVLA's real vision shapes on SM87.

The FP16 test keeps softmax arithmetic FP32 via explicit casts in the TRT graph.
Graph replay timings use persistent input/output buffers and exclude compilation,
allocation and copies. This is an operator experiment, not policy throughput.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import tensorrt as trt
import torch
import triton

from bench.vendor.groot_trt import _cuda
from bench.vendor.trt_device import Device
from vendor.lab_softmax import _softmax_kernel, _softmax_online_kernel

p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
assert torch.cuda.get_device_capability()==(8,7), 'This experiment targets the Orin Nano'
shape=(1,12,1024,1024)
rows,cols=int(np.prod(shape[:-1])),shape[-1]
stream=torch.cuda.Stream()
d=Device.__new__(Device)
d.cu=_cuda();d.stream=ctypes.c_void_p(stream.cuda_stream);d.capturing=False
logger=trt.Logger(trt.Logger.ERROR)
results=[]
engines=[]
contexts=[]
graphs=[]


def timed(fn,repeat=9,batch=50):
    for _ in range(5):fn()
    d.sync()
    graph=d.capture(fn)
    graphs.append(graph)
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    durations=[]
    for _ in range(repeat):
        start.record(stream)
        for _ in range(batch):d.launch(graph)
        end.record(stream);end.synchronize()
        durations.append(start.elapsed_time(end)/batch)
    return dict(median_ms=float(np.median(durations)),p25_ms=float(np.percentile(durations,25)),
                p75_ms=float(np.percentile(durations,75)),batches=durations)


with torch.cuda.stream(stream):
    torch.manual_seed(1234)
    for dtype in [torch.float32,torch.float16]:
        x=torch.randn(shape,device='cuda',dtype=dtype)
        x[...,cols//2:]-=10000 # nontrivial mask-like values, at least one valid key per row
        y=torch.empty_like(x)
        ref=torch.softmax(x.float(),dim=-1).to(dtype)
        saved=x.clone()
        builder=trt.Builder(logger)
        net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        inp=net.add_input('x',trt.float32 if dtype==torch.float32 else trt.float16,shape)
        val=inp if dtype==torch.float32 else net.add_cast(inp,trt.float32).get_output(0)
        soft=net.add_softmax(val);soft.axes=1<<(len(shape)-1)
        val=soft.get_output(0)
        if dtype==torch.float16:val=net.add_cast(val,trt.float16).get_output(0)
        val.name='out';net.mark_output(val)
        config=builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,64<<20)
        config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        plan=builder.build_serialized_network(net,config)
        assert plan is not None
        runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(plan)
        context=engine.create_execution_context()
        context.set_tensor_address('x',x.data_ptr());context.set_tensor_address('out',y.data_ptr())
        engines.append((runtime,engine));contexts.append(context)
        def run_trt():
            assert context.execute_async_v3(stream.cuda_stream)
        run_trt();d.sync()
        tol=dict(rtol=1.3e-6,atol=1e-6) if dtype==torch.float32 else dict(rtol=1e-3,atol=1e-3)
        torch.testing.assert_close(y.float(),ref.float(),**tol)
        baseline=timed(run_trt)
        inspector=engine.create_engine_inspector()
        baseline['layers']=json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        tests=[]
        for variant,warps,tile in [('row',w,1024) for w in [4,8,16]]+ [('online',4,256),('online',4,512)]:
            def run_kernel():
                if variant=='row':
                    _softmax_kernel[(rows,)](x,y,cols,cols,cols,BLOCK_N=1024,num_warps=warps)
                else:
                    _softmax_online_kernel[(rows,)](x,y,cols,cols,cols,TILE_N=tile,num_warps=warps)
            t0=time.perf_counter();run_kernel();d.sync();compile_s=time.perf_counter()-t0
            torch.testing.assert_close(y.float(),ref.float(),**tol)
            max_abs=float((y.float()-ref.float()).abs().max().item())
            # Exercise stable large logits and row normalization in addition to masked random logits.
            x.fill_(30000);run_kernel();d.sync()
            torch.testing.assert_close(y.float().sum(-1),torch.ones(shape[:-1],device='cuda'),rtol=2e-3,atol=2e-3)
            x.copy_(saved)
            ref=torch.softmax(x.float(),dim=-1).to(dtype)
            run_kernel();d.sync();torch.testing.assert_close(y.float(),ref.float(),**tol)
            timings=timed(run_kernel)
            tests.append(dict(variant=variant,warps=warps,tile=tile,compile_s=compile_s,max_abs=max_abs,
                              speedup_vs_trt=baseline['median_ms']/timings['median_ms'],**timings))
            print(str(dtype),variant,warps,tile,timings['median_ms'],flush=True)
        results.append(dict(dtype=str(dtype),shape=shape,accumulation='FP32',baseline=baseline,triton=tests))
        for graph in graphs:d.destroy_graph(graph)
        graphs.clear()
        del x,y,ref,saved,context,engine,runtime,builder,net,config,plan
        contexts.clear();engines.clear()
        torch.cuda.empty_cache()

result=dict(status='PASS',scope='standalone softmax operator; CUDA graph replay; no policy speed claim',
            device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
            torch=torch.__version__,torch_cuda=torch.version.cuda,triton=triton.__version__,tensorrt=trt.__version__,
            lab_source_sha256=hashlib.sha256((Path(__file__).parent/'vendor/lab_softmax.py').read_bytes()).hexdigest(),
            lab_license='MIT, Copyright (c) 2026 Jeremy Gracey; vendor/LICENSE.triton-kernel-lab',results=results)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2))
print('PASS ->',a.out)
