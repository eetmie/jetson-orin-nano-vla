"""Check/tune an attention kernel at the real Nano vision layout before integration.

The Torch expression is only a numerical oracle. Timings are captured kernel
replays with persistent buffers, excluding imports, compilation, and copies.
"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    assert torch.cuda.get_device_capability() == (8, 7)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(20261010)
    shape = (1, 12, 1024, 64)
    q = torch.randn(shape, device='cuda', dtype=torch.float16)*0.35
    kt = torch.randn((1, 12, 64, 1024), device='cuda', dtype=torch.float16)*0.35
    v = torch.randn(shape, device='cuda', dtype=torch.float16)
    mask = torch.zeros((1, 1, 1024, 1024), device='cuda', dtype=torch.float16)
    out = torch.empty_like(q)
    cases = []
    for case in ['unmasked', 'partial_mask', 'fully_masked_rows', 'infinities']:
        mask.zero_()
        if case in ['partial_mask', 'fully_masked_rows']:
            mask[..., :512] = -float('inf')
        if case == 'fully_masked_rows':
            mask[..., 0, :] = -float('inf')
        if case == 'infinities':
            mask[..., 0, 0] = float('inf')
            mask[..., 1, 0] = float('nan')
        logits = ((q@kt).half()+mask).half()
        prob = torch.softmax(logits.float(), -1).half()
        prob = torch.where(torch.isnan(prob), 0, prob)
        reference = prob@v
        for bm, bn in [(32, 64), (64, 64), (64, 128), (128, 64)]:
            attention_aot[(triton.cdiv(1024, bm), 12)](
                q, kt, v, mask, 1024, out, BLOCK_M=bm, BLOCK_N=bn, D=64,
                num_warps=4, num_stages=2)
            torch.cuda.synchronize()
            assert torch.isfinite(out).all(), (case, bm, bn)
            torch.testing.assert_close(out.float(), reference.float(), rtol=2e-2, atol=3e-4)
            if case in ['fully_masked_rows', 'infinities']:
                assert (out[..., 0, :]==0).all()
            cases.append(dict(case=case, block_m=bm, block_n=bn,
                              max_abs=float((out.float()-reference.float()).abs().max()), status='PASS'))
            print('CHECK', cases[-1], flush=True)
    mask.zero_()
    device = Device.__new__(Device)
    device.cu = _cuda()
    device.stream = ctypes.c_void_p(stream.cuda_stream)
    device.capturing = False
    timings = []
    for bm, bn in [(32, 64), (64, 64), (64, 128), (128, 64)]:
        def run():
            return attention_aot[(triton.cdiv(1024, bm), 12)](
                q, kt, v, mask, 1024, out, BLOCK_M=bm, BLOCK_N=bn, D=64,
                num_warps=4, num_stages=2)
        compiled = run()
        torch.cuda.synchronize()
        assert getattr(compiled.metadata, 'global_scratch_size', 0)==0
        graph = device.capture(run)
        try:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            batches = []
            for _ in range(9):
                start.record(stream)
                for _ in range(50):
                    device.launch(graph)
                end.record(stream)
                end.synchronize()
                batches.append(start.elapsed_time(end)/50)
            timings.append(dict(block_m=bm, block_n=bn, warps=4, stages=2,
                                median_ms=float(np.median(batches)), batches_ms=batches,
                                shared_memory_bytes=compiled.metadata.shared))
            print('TIMING', timings[-1], flush=True)
        finally:
            device.destroy_graph(graph)
    report = dict(status='PASS', shape=shape, torch=torch.__version__, triton=triton.__version__,
                  capability=torch.cuda.get_device_capability(), cases=cases, timings=timings,
                  caveat='Online probabilities round before global normalization; full-action gates required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
