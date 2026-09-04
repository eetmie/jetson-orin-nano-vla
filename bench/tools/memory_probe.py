#!/usr/bin/env python3
"""Where does X-VLA's resident memory actually go, and which knob moves it?

The build wall is solved (see notes/split_design.md); the remaining problem is that 12
resident TRT sessions leave nothing for the control stack on a 7.4 GB board. This loads
the engines one at a time under a named configuration and reports the marginal cost of
each, so the fix is chosen from measurement rather than from guessing which ORT option
sounds expensive.

Run one config per process -- ORT and CUDA allocate lazily and never fully hand memory
back, so comparing configs inside a single process measures the order they ran in.

    python tools/memory_probe.py --config baseline
    python tools/memory_probe.py --compare        # spawns one subprocess per config
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Each config is (env overrides, session-option overrides) applied before any session
# is created. Names are what --compare reports.
CONFIGS: dict[str, dict] = {
    "baseline": {},
    "no_cuda_ep": {"env": {"TRT_DROP_CUDA_EP": "1"}},
    "no_arena": {"session": {"enable_cpu_mem_arena": False}},
    "no_arena_no_pattern": {"session": {"enable_cpu_mem_arena": False,
                                        "enable_mem_pattern": False}},
    "no_cuda_ep_no_arena": {"env": {"TRT_DROP_CUDA_EP": "1"},
                            "session": {"enable_cpu_mem_arena": False}},
}


def available_gb() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def rss_gb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def run_config(name: str, split_dir: Path, cache_dir: str, precision: str) -> dict:
    cfg = CONFIGS[name]
    for k, v in cfg.get("env", {}).items():
        os.environ[k] = v

    import onnxruntime as ort

    from bench.vendor.xvla_split_ort import build_providers

    bundle = json.loads((split_dir / "bundle.json").read_text())
    providers = build_providers(cache_dir, precision)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    for k, v in cfg.get("session", {}).items():
        setattr(so, k, v)

    base_rss, base_avail = rss_gb(), available_gb()
    print(f"\n[{name}]  start rss {base_rss:.2f} GB, available {base_avail:.2f} GB")
    print(f"{'engine':16s} {'params':>8s} {'d_rss':>8s} {'rss':>8s} {'avail':>8s} {'load':>7s}"
          f"  providers")

    from bench.vendor.xvla_split_ort import ep_context_path

    sessions = []
    rows = []
    prev = base_rss
    for g in bundle["graphs"]:
        # Match the runtime: prefer the EPContext stand-in when one has been dumped,
        # otherwise this measures a load path the policy does not actually use.
        model_path = split_dir / g["file"]
        ctx = ep_context_path(split_dir, g["file"])
        if ctx.exists():
            model_path = ctx
        t0 = time.time()
        s = ort.InferenceSession(
            str(model_path), sess_options=so, providers=providers
        )
        load_s = time.time() - t0
        sessions.append(s)
        now = rss_gb()
        # Which EP actually took the graph: a CPU/CUDA fallback here would mean the
        # engine is not doing what we think, and would also explain extra memory.
        eps = "+".join(p.replace("ExecutionProvider", "") for p in s.get_providers())
        rows.append({"name": g["name"], "params": g["params"], "d_rss_gb": now - prev,
                     "rss_gb": now, "load_s": load_s, "providers": eps})
        print(f"{g['name']:16s} {g['params'] / 1e6:7.1f}M {now - prev:7.2f}G {now:7.2f}G "
              f"{available_gb():7.2f}G {load_s:6.1f}s  {eps}")
        prev = now

    total_params = sum(g["params"] for g in bundle["graphs"])
    result = {
        "config": name,
        "start_rss_gb": round(base_rss, 2),
        "loaded_rss_gb": round(rss_gb(), 2),
        "session_cost_gb": round(rss_gb() - base_rss, 2),
        "available_after_gb": round(available_gb(), 2),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "total_params": total_params,
        "fp32_weights_gb": round(total_params * 4 / 1e9, 2),
        "engines": rows,
    }
    print(f"\n  sessions cost {result['session_cost_gb']:.2f} GB for "
          f"{result['fp32_weights_gb']:.2f} GB of FP32 weights "
          f"({result['session_cost_gb'] / result['fp32_weights_gb']:.2f}x)")
    print(f"  rss {result['loaded_rss_gb']:.2f} GB, available {result['available_after_gb']:.2f} GB")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", type=Path, default=Path.home() / "bundles" / "xvla-base-split")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--config", default="baseline", choices=sorted(CONFIGS))
    ap.add_argument("--compare", action="store_true",
                    help="run every config, one subprocess each")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "memory_probe_results.json")
    args = ap.parse_args()

    cache_dir = args.cache_dir or str(args.split_dir / "trt_cache")

    if not args.compare:
        res = run_config(args.config, args.split_dir, cache_dir, args.precision)
        print("MEMPROBE_RESULT " + json.dumps(res))
        return

    results = []
    for name in sorted(CONFIGS):
        proc = subprocess.run(
            [sys.executable, __file__, "--config", name, "--split-dir", str(args.split_dir),
             "--cache-dir", cache_dir, "--precision", args.precision],
            capture_output=True, text=True,
        )
        got = None
        for line in proc.stdout.splitlines():
            if line.startswith("MEMPROBE_RESULT "):
                got = json.loads(line[len("MEMPROBE_RESULT "):])
        if got is None:
            print(f"[{name}] FAILED (exit {proc.returncode})")
            for line in (proc.stderr or proc.stdout).strip().splitlines()[-8:]:
                print(f"   {line}")
            continue
        results.append(got)
        print(f"[{name}] sessions {got['session_cost_gb']:.2f} GB | rss "
              f"{got['loaded_rss_gb']:.2f} GB | available after "
              f"{got['available_after_gb']:.2f} GB", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    if results:
        best = min(results, key=lambda r: r["loaded_rss_gb"])
        base = next((r for r in results if r["config"] == "baseline"), None)
        print(f"\nlowest: {best['config']} at {best['loaded_rss_gb']:.2f} GB rss")
        if base and best["config"] != "baseline":
            print(f"  saves {base['loaded_rss_gb'] - best['loaded_rss_gb']:.2f} GB "
                  f"vs baseline")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
