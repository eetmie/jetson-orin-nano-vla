"""Compare the expert gated projection against an isolated TensorRT fusion.

Synthetic weights, actual expert shapes; CUDA graph replay removes Python overhead.
An operator gain must pass complete-policy measurements before acceptance.
"""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import numpy as np
import tensorrt as trt
import torch
import triton
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.vendor.groot_trt import _cuda
from bench.vendor.trt_device import Device
from ffn_kernel import gated_projection, packed_gated_projection

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--extended', action='store_true', help='Test wider tiles and pipeline depths')
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
torch.manual_seed(20261010)
assert torch.cuda.get_device_capability() == (8, 7)
stream = torch.cuda.Stream()
torch.cuda.set_stream(stream)
x = torch.randn((1, 50, 720), device='cuda', dtype=torch.float16)
g = torch.randn((720, 2048), device='cuda', dtype=torch.float16)*.04
u = torch.randn_like(g)*.04
ref_gate = x@g
ref = ((ref_gate*torch.sigmoid(ref_gate))*(x@u)).half()
out = torch.empty_like(ref)
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
net = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
inp = net.add_input('x', trt.float16, (1, 50, 720))
gate_w = net.add_constant((1, 720, 2048), g.cpu().numpy()[None]).get_output(0)
up_w = net.add_constant((1, 720, 2048), u.cpu().numpy()[None]).get_output(0)
gate = net.add_matrix_multiply(inp, trt.MatrixOperation.NONE, gate_w, trt.MatrixOperation.NONE).get_output(0)
up = net.add_matrix_multiply(inp, trt.MatrixOperation.NONE, up_w, trt.MatrixOperation.NONE).get_output(0)
sig = net.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0)
silu = net.add_elementwise(gate, sig, trt.ElementWiseOperation.PROD).get_output(0)
result = net.add_elementwise(silu, up, trt.ElementWiseOperation.PROD).get_output(0)
result.name = 'y'
net.mark_output(result)
config = builder.create_builder_config()
config.builder_optimization_level = 2
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64<<20)
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
plan = builder.build_serialized_network(net, config)
assert plan is not None
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(plan)
context = engine.create_execution_context()
context.set_tensor_address('x', x.data_ptr())
context.set_tensor_address('y', out.data_ptr())
device = Device.__new__(Device)
device.cu = _cuda()
device.stream = ctypes.c_void_p(stream.cuda_stream)
device.capturing = False

def measure(fn):
    for _ in range(10):
        fn()
    stream.synchronize()
    max_abs = float((out-ref).float().abs().max())
    cosine = float(torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0))
    graph = device.capture(fn)
    batches = []
    try:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(9):
            start.record(stream)
            for _ in range(100):
                device.launch(graph)
            end.record(stream)
            end.synchronize()
            batches.append(start.elapsed_time(end)/100)
    finally:
        device.destroy_graph(graph)
    assert torch.isfinite(out).all() and cosine >= .999
    return dict(median_ms=float(np.median(batches)), batches_ms=batches, max_abs=max_abs, cosine=cosine)

def trt_run():
    assert context.execute_async_v3(stream.cuda_stream)

baseline = measure(trt_run)
print('TRT', baseline, flush=True)
rows = []
packed = torch.stack([g,u],dim=-1).reshape(720,4096).contiguous()
for bm, bn, bk, warps, stages in ([] if args.extended else [(16,64,32,4,2), (32,64,32,4,2), (32,128,32,4,2),
                                 (64,64,32,4,2), (16,128,64,4,2), (32,128,64,4,2),
                                 (32,64,64,4,3), (64,128,32,8,2)]):
    def run():
        gated_projection[(triton.cdiv(50,bm), triton.cdiv(2048,bn))](
            x,g,u,out,M=50,N=2048,K=720,BM=bm,BN=bn,BK=bk,
            num_warps=warps,num_stages=stages)
    try:
        row = dict(bm=bm,bn=bn,bk=bk,warps=warps,stages=stages,**measure(run))
    except Exception as exc:
        row = dict(bm=bm,bn=bn,bk=bk,warps=warps,stages=stages,error=str(exc))
    print('TRITON', row, flush=True)
    rows.append(row)
for bm,bn,bk,warps,stages in ([(64,128,32,4,2), (64,256,32,8,2), (64,256,64,8,2),
                              (64,128,64,4,2), (64,128,32,4,3), (64,128,32,4,4),
                              (64,128,64,8,3), (64,64,32,4,3)] if args.extended else
                            [(32,64,32,4,2), (64,64,32,4,2), (64,128,32,8,2),
                             (64,128,64,8,2), (32,128,64,4,2), (64,64,64,4,2),
                             (32,64,64,4,3), (64,128,32,4,2)]):
    def run_packed():
        packed_gated_projection[(triton.cdiv(50,bm),triton.cdiv(2048,bn))](
            x,packed,out,M=50,N=2048,K=720,BM=bm,BN=bn,BK=bk,
            num_warps=warps,num_stages=stages)
    try:
        row = dict(layout='interleaved_gate_up',bm=bm,bn=bn,bk=bk,warps=warps,stages=stages,**measure(run_packed))
    except Exception as exc:
        row = dict(layout='interleaved_gate_up',bm=bm,bn=bn,bk=bk,warps=warps,stages=stages,error=str(exc))
    print('PACKED TRITON',row,flush=True)
    rows.append(row)
report = dict(scope='Isolated HALF gated projection, synthetic weights, actual expert shape; CUDA graph replay',
              shape=dict(m=50,n=2048,k=720),triton=triton.__version__,tensorrt=trt.__version__,
              baseline=baseline,candidates=rows,
              engine_layers=json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)))
args.out.write_text(json.dumps(report,indent=2))
