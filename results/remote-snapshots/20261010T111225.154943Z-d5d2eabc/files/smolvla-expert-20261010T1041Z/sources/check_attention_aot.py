"""Isolate AOT compiler attributes from TensorRT integration at fixed SM87 shapes."""
import argparse
import ctypes
import json
from pathlib import Path
import sys

import numpy as np
import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.vendor.groot_trt import _cuda
from bench.vendor.trt_device import Device
from attention_kernel import attention_aot

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
assert torch.cuda.get_device_capability() == (8, 7)
stream = torch.cuda.Stream()
torch.cuda.set_stream(stream)
torch.manual_seed(20261010)
q = torch.randn((1, 12, 1024, 64), device='cuda', dtype=torch.float16)*0.35
kt = torch.randn((1, 12, 64, 1024), device='cuda', dtype=torch.float16)*0.35
v = torch.randn_like(q)
mask = torch.zeros((1, 1, 1024, 1024), device='cuda', dtype=torch.float16)
out, ref = torch.empty_like(q), torch.empty_like(q)
attention_aot[(8, 12)](q, kt, v, mask, 1024, ref, BLOCK_M=128, BLOCK_N=64, D=64, num_warps=4, num_stages=2)
torch.cuda.synchronize()
device = Device.__new__(Device)
device.cu = _cuda()
device.stream = ctypes.c_void_p(stream.cuda_stream)
device.capturing = False
results = []
for aligned in [False, True]:
    attrs = {(i,): [('tt.divisibility', 16)] for i in range(6)} if aligned else {}
    source = triton.compiler.ASTSource(attention_aot,
        signature={'Q':'*fp16','Kt':'*fp16','V':'*fp16','Mask':'*fp16','n_ctx':'i32','O':'*fp16'},
        constexprs={'BLOCK_M':128, 'BLOCK_N':64, 'D':64}, attrs=attrs)
    kernel = triton.compile(source, options={'num_warps':4, 'num_stages':2})
    def run():
        kernel[(8, 12, 1)](q, kt, v, mask, 1024, out)
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)
    graph = device.capture(run)
    try:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        batches = []
        for _ in range(9):
            start.record(stream)
            for _ in range(25):
                device.launch(graph)
            end.record(stream)
            end.synchronize()
            batches.append(start.elapsed_time(end)/25)
        row = dict(alignment16=aligned, median_ms=float(np.median(batches)), batches_ms=batches,
                   max_abs=float((out-ref).float().abs().max()), shared_bytes=kernel.metadata.shared)
        print('PASS', row, flush=True)
        results.append(row)
    finally:
        device.destroy_graph(graph)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(dict(status='PASS', results=results, triton=triton.__version__), indent=2))
