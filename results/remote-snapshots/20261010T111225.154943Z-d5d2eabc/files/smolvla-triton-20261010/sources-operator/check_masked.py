"""Check the fused kernel against the exported FP16 Add / FP32 softmax semantics.

Torch is a numerical oracle here, not a policy timing baseline. Run on SM87 in
the build environment. Mask rows broadcast across the twelve attention heads.
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
from aot_kernel import masked_softmax_aot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (8, 7)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    torch.manual_seed(20261010)
    shape = (1, 12, 1024, 1024)
    cols = shape[-1]
    cases = []
    for case in ['random', 'partial_and_fully_masked', 'large_rounding', 'infinities']:
        logits = torch.randn(shape, dtype=torch.float16, device='cuda')
        mask = torch.zeros((1, 1, cols, cols), dtype=torch.float16, device='cuda')
        if case == 'partial_and_fully_masked':
            mask[..., 512:] = -float('inf')
            mask[..., 0, :] = -float('inf')
        elif case == 'large_rounding':
            logits.mul_(2).add_(30000)
            mask[..., ::2] = -29984
        elif case == 'infinities':
            logits[..., 0, 0] = float('inf')
            mask[..., 1, :] = -float('inf')
        # The HALF addition must round before the FP32 softmax.
        reference = torch.softmax((logits + mask).float(), dim=-1).half()
        reference = torch.where(torch.isnan(reference), 0, reference)
        actual = torch.empty_like(logits)
        masked_softmax_aot[(12 * cols,)](
            logits, mask, cols, cols, actual, BLOCK_N=cols, num_warps=4)
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all(), case
        torch.testing.assert_close(actual.float(), reference.float(), rtol=1e-3, atol=1e-5)
        cases.append(dict(case=case, max_abs=float((actual.float()-reference.float()).abs().max()),
                          reference_zero_rows=int((reference.sum(-1) == 0).sum()), status='PASS'))
        print(case, cases[-1], flush=True)
    device = Device.__new__(Device)
    device.cu = _cuda()
    device.stream = ctypes.c_void_p(stream.cuda_stream)
    device.capturing = False
    timings = []
    logits.normal_()
    mask.zero_()
    mask[..., 512:] = -float('inf')
    reference = torch.softmax((logits + mask).float(), dim=-1).half()
    for warps in [1, 2, 4, 8]:
        def run():
            masked_softmax_aot[(12 * cols,)](
                logits, mask, cols, cols, actual, BLOCK_N=cols, num_warps=warps)
        for _ in range(5):
            run()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.float(), reference.float(), rtol=1e-3, atol=1e-5)
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
            timings.append(dict(warps=warps, median_ms=float(np.median(batches)), batches_ms=batches))
            print('TIMING', timings[-1], flush=True)
        finally:
            device.destroy_graph(graph)
    report = dict(status='PASS', shape=shape, mask_shape=(1, 1, cols, cols),
                  capability=torch.cuda.get_device_capability(), torch=torch.__version__,
                  triton=triton.__version__, cases=cases, graph_replay_timings=timings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
