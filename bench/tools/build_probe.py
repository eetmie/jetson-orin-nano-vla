#!/usr/bin/env python3
"""Calibrate the TensorRT build-memory curve on this board.

The SmolVLA deploy hit a build-time wall that had nothing to do with graph size: TRT
imports weights as FP32 working copies, so the peak tracks the *weight slice* an engine
carries. X-VLA is ~2x SmolVLA, so before exporting anything we need the real curve --
peak host memory as a function of parameters-per-engine -- to choose where to cut.

Export and build run in SEPARATE subprocesses, and the build child never imports torch.
This matters: a first version of this probe traced the ONNX and built the engine in one
process, and the PyTorch side alone put it near the physical ceiling, so every "peak" was
really the box saturating rather than what TRT needed. The deploy pipeline exports and
builds separately too, so measuring them separately is also the honest model of it.

Weights are random on purpose -- build peak depends on shapes and weight volume, not
values -- so this runs without the checkpoint.

    python tools/build_probe.py --blocks 4 8 12 24
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# X-VLA policy-transformer geometry (config.json of lerobot/xvla-base).
HIDDEN = 1024
NUM_HEADS = 16
MLP_RATIO = 4.0
SEQ_LEN = 262  # 30 action + 100 vlm + 100 aux + 32 soft prompts


def _meminfo_available_kb() -> int:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return -1


class MemorySampler(threading.Thread):
    """Track the system-wide available-memory low-water mark during a phase.

    Jetson GPU allocations are unified and do not all land in the child's RSS, so the
    system-wide dip is the number that actually predicts an OOM.
    """

    def __init__(self, interval: float = 0.2) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.baseline_kb = _meminfo_available_kb()
        self.min_available_kb = self.baseline_kb
        # not `_stop`: threading.Thread already uses that name for an internal method
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            self.min_available_kb = min(self.min_available_kb, _meminfo_available_kb())

    def stop(self) -> dict[str, float]:
        self._halt.set()
        self.join(timeout=2.0)
        return {
            "baseline_available_gb": round(self.baseline_kb / 1e6, 2),
            "min_available_gb": round(self.min_available_kb / 1e6, 2),
            "consumed_gb": round((self.baseline_kb - self.min_available_kb) / 1e6, 2),
        }


# --------------------------------------------------------------------------------------
# child A: export an N-block stack to ONNX (torch lives only here)
# --------------------------------------------------------------------------------------

EXPORT_CHILD = r'''
import json, sys, resource
from pathlib import Path
import torch, torch.nn as nn

blocks, onnx_out = int(sys.argv[1]), Path(sys.argv[2])
HIDDEN, NUM_HEADS, MLP_RATIO, SEQ_LEN = {hidden}, {heads}, {mlp}, {seq}
# the real block, so the probed graph matches what the denoise engines will contain
from lerobot.policies.xvla.soft_transformer import TransformerBlock

class Stack(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(HIDDEN, NUM_HEADS, mlp_ratio=MLP_RATIO) for _ in range(n)]
        )
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

model = Stack(blocks).eval()
n_params = sum(p.numel() for p in model.parameters())
onnx_out.parent.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    torch.onnx.export(
        model, (torch.zeros(1, SEQ_LEN, HIDDEN),), str(onnx_out),
        input_names=["x"], output_names=["y"], opset_version=17, dynamo=False,
    )
print("EXPORT_RESULT " + json.dumps({{
    "params": n_params,
    "fp32_weights_gb": round(n_params * 4 / 1e9, 3),
    "onnx_mb": round(onnx_out.stat().st_size / 1e6, 1),
    "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
}}))
'''

# --------------------------------------------------------------------------------------
# child B: build the TRT engine from that ONNX -- deliberately no torch import
# --------------------------------------------------------------------------------------

BUILD_CHILD = r'''
import json, os, sys, resource, time
from pathlib import Path
import numpy as np
import onnxruntime as ort

onnx_path, cache, precision = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
HIDDEN, SEQ_LEN = {hidden}, {seq}

cache.mkdir(parents=True, exist_ok=True)
trt_opts = {{
    "device_id": 0,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": str(cache),
    "trt_max_workspace_size": int(os.environ.get("TRT_WORKSPACE_MB", "512")) * 1024 * 1024,
    "trt_builder_optimization_level": int(os.environ.get("TRT_OPT_LEVEL", "2")),
    "trt_layer_norm_fp32_fallback": True,
}}
if precision == "fp16":
    trt_opts["trt_fp16_enable"] = True

providers = [("TensorrtExecutionProvider", trt_opts)]
if not os.environ.get("TRT_DROP_CUDA_EP"):
    providers.append("CUDAExecutionProvider")
providers.append("CPUExecutionProvider")

so = ort.SessionOptions()
so.log_severity_level = 3
t0 = time.time()
sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
build_s = time.time() - t0
build_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

x = np.zeros((1, SEQ_LEN, HIDDEN), dtype=np.float32)
sess.run(None, {{"x": x}})                      # warm up
t0 = time.time()
for _ in range(10):
    sess.run(None, {{"x": x}})
infer_ms = (time.time() - t0) / 10 * 1000

print("BUILD_RESULT " + json.dumps({{
    "build_s": round(build_s, 1),
    "build_peak_rss_gb": round(build_rss, 2),
    "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
    "infer_ms": round(infer_ms, 1),
    "engine_cache_mb": round(
        sum(f.stat().st_size for f in cache.glob("*") if f.is_file()) / 1e6, 1
    ),
    "providers": sess.get_providers(),
}}))
'''


def _run_phase(cmd: list[str], tag: str, env: dict) -> tuple[dict | None, dict]:
    sampler = MemorySampler()
    sampler.start()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    mem = sampler.stop()
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith(f"{tag} "):
            result = json.loads(line[len(tag) + 1 :])
    if result is None:
        print(f"  !! {tag.lower()} failed (exit {proc.returncode})")
        for line in (proc.stderr or proc.stdout).strip().splitlines()[-10:]:
            print(f"     {line}")
    return result, mem


def run_probe(blocks: int, precision: str, workdir: Path, keep: bool) -> dict:
    onnx_path = workdir / f"probe_{blocks}blk.onnx"
    cache_dir = workdir / f"cache_{blocks}blk_{precision}"
    fmt = {"hidden": HIDDEN, "heads": NUM_HEADS, "mlp": MLP_RATIO, "seq": SEQ_LEN}

    exp_src = workdir / f"_export_{blocks}.py"
    exp_src.write_text(EXPORT_CHILD.format(**fmt))
    bld_src = workdir / f"_build_{blocks}.py"
    bld_src.write_text(BUILD_CHILD.format(**fmt))

    env = {**os.environ, "TRT_DROP_CUDA_EP": "1", "TRT_WORKSPACE_MB": "512", "TRT_OPT_LEVEL": "2"}
    out: dict = {"blocks": blocks, "precision": precision}

    t0 = time.time()
    exp, exp_mem = _run_phase(
        [sys.executable, str(exp_src), str(blocks), str(onnx_path)],
        "EXPORT_RESULT", env,
    )
    if exp is None:
        exp_src.unlink(missing_ok=True)
        bld_src.unlink(missing_ok=True)
        return {**out, "failed": "export", "export_consumed_gb": exp_mem["consumed_gb"]}
    # rename before merging: the build child reports its own `peak_rss_gb` and would
    # otherwise silently overwrite the export's
    out.update({k: v for k, v in exp.items() if k != "peak_rss_gb"})
    out["export_peak_rss_gb"] = exp["peak_rss_gb"]
    out["export_consumed_gb"] = exp_mem["consumed_gb"]
    out["export_s"] = round(time.time() - t0, 1)

    bld, bld_mem = _run_phase(
        [sys.executable, str(bld_src), str(onnx_path), str(cache_dir), precision],
        "BUILD_RESULT", env,
    )
    exp_src.unlink(missing_ok=True)
    bld_src.unlink(missing_ok=True)
    if not keep:
        onnx_path.unlink(missing_ok=True)
    if bld is None:
        return {**out, "failed": "build", "build_consumed_gb": bld_mem["consumed_gb"]}
    out.update(bld)
    out["build_consumed_gb"] = bld_mem["consumed_gb"]
    out["build_min_available_gb"] = bld_mem["min_available_gb"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, nargs="+", default=[4, 8, 12, 24])
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--workdir", type=Path, default=REPO / "results" / "probe")
    ap.add_argument("--keep-onnx", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "build_probe_results.json")
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    print(f"TRT build-memory probe  (precision={args.precision}, seq_len={SEQ_LEN}, hidden={HIDDEN})")
    print(f"system available at start: {_meminfo_available_kb() / 1e6:.2f} GB")
    print("export and build are separate processes; the build child never imports torch\n")

    results = []
    for blocks in args.blocks:
        print(f"[{blocks:2d} blocks]", flush=True)
        res = run_probe(blocks, args.precision, args.workdir, args.keep_onnx)
        results.append(res)
        if res.get("failed"):
            print(f"  FAILED in {res['failed']} phase\n", flush=True)
            continue
        print(
            f"  {res['params'] / 1e6:6.1f}M params ({res['fp32_weights_gb']:.2f} GB fp32)\n"
            f"  export: peak_rss {res['export_peak_rss_gb']:.2f} GB, consumed "
            f"{res['export_consumed_gb']:.2f} GB, {res['export_s']:.0f}s\n"
            f"  build : peak_rss {res['build_peak_rss_gb']:.2f} GB, consumed "
            f"{res['build_consumed_gb']:.2f} GB, {res['build_s']:.0f}s, "
            f"engine {res['engine_cache_mb']:.0f} MB\n"
            f"  infer : {res['infer_ms']:.1f} ms\n",
            flush=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))

    ok = [r for r in results if not r.get("failed")]
    if len(ok) >= 2:
        lo, hi = ok[0], ok[-1]
        dw = hi["fp32_weights_gb"] - lo["fp32_weights_gb"]
        if dw > 0:
            for label, key in (("build peak RSS", "build_peak_rss_gb"),
                               ("system consumed", "build_consumed_gb")):
                slope = (hi[key] - lo[key]) / dw
                floor = lo[key] - slope * lo["fp32_weights_gb"]
                print(f"fit ({label}): {floor:.2f} GB + {slope:.2f} x (fp32 weight GB)")
                if slope > 0:
                    budget = _meminfo_available_kb() / 1e6
                    print(f"     at {budget:.1f} GB available -> "
                          f"max ~{(budget - floor) / slope:.2f} GB fp32 weights per engine")
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
