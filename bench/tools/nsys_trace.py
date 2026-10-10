#!/usr/bin/env python3
"""A short Nsight Systems trace of one trt-split model, and a summary of where its time goes.

Every engine call and every inference is an NVTX range (through CUDA's own
libnvtx3interop, so nothing is installed), and capture covers only the measured loop:
loading, engine builds and warmup run before cudaProfilerStart.

    nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop \\
        --sample=none --cpuctxsw=none --cuda-graph-trace=node -o traces/xvla-base \\
        .venv-ort/bin/python -m bench.tools.nsys_trace --model xvla-base \\
        --bundle ~/bundles/xvla-base-split
    nsys export --type sqlite -o traces/xvla-base.sqlite traces/xvla-base.nsys-rep
    python -m bench.tools.nsys_trace --summarize traces/xvla-base.sqlite

The summary splits each stage's wall time into GPU kernels, copies, the rest of the
engine call (launch, synchronize, Python) and host glue between calls, and divides the
engine's size by its kernel time: a stage that reads its weights at close to the
board's ~102 GB/s is bandwidth-bound, one far below it is not. Traced latency is a
little above an untraced run's; read the split, not the total.

`--chain host` (the default here) labels every engine call. `--chain graph` traces have
no per-engine ranges, only kernels, and nsys shows those only with
`--cuda-graph-trace=node`.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


class _Nvtx:
    def __init__(self):
        self.lib = None
        for name in ("libnvtx3interop.so.1", "/usr/local/cuda/lib64/libnvtx3interop.so.1"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                pass

    def push(self, text: str) -> None:
        if self.lib:
            self.lib.nvtxRangePushA(text.encode())

    def pop(self) -> None:
        if self.lib:
            self.lib.nvtxRangePop()


def trace(a) -> None:
    from bench.cli import Resolved, trt_backend
    from bench.obs import make_obs
    from bench.vendor.groot_trt import _cuda

    bundle = Path(a.bundle).expanduser()
    r = Resolved(a, bundle)
    be = trt_backend(a, r, bundle)
    be.load()
    nv = _Nvtx()
    run = be.engines.run

    def traced(name, feeds):
        nv.push(f"engine {name}")
        try:
            return run(name, feeds)
        finally:
            nv.pop()

    be.engines.run = traced
    if hasattr(be, "policy"):                     # SmolVLA binds run at construction
        be.policy.run = traced
    obs = make_obs("synthetic", r.task, r.chunk_size, r.state_dim,
                   max_action_dim=r.noise_width, seed=a.seed, n_views=r.views,
                   noise_distribution=r.noise_distribution)
    # Rendered before capture, as the runner does: a synthetic frame costs tens of ms.
    ring = [obs[i] for i in range(a.warmup + a.iters)]
    for i in range(a.warmup):
        be.infer(ring[i])
    cache = Path(be.cache_dir).expanduser()
    side = {"model": a.model, "iters": a.iters,
            "engine_mb": {p.stem: round(p.stat().st_size / 1e6, 1)
                          for p in cache.glob("*.engine")},
            "loop_ms": []}
    _cuda().cudaProfilerStart()
    for i in range(a.iters):
        nv.push(f"infer {i}")
        t0 = time.perf_counter()
        be.infer(ring[a.warmup + i])
        nv.pop()
        side["loop_ms"].append(round((time.perf_counter() - t0) * 1e3, 2))
    _cuda().cudaProfilerStop()
    out = Path(a.sidecar or f"{a.model}.trace.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(side, indent=1))
    print(f"traced {a.iters} inferences; wrote {out}")


def _ranges(db, prefix: str) -> list[tuple[int, int, str]]:
    q = """SELECT e.start, e.end, COALESCE(e.text, s.value) FROM NVTX_EVENTS e
           LEFT JOIN StringIds s ON e.textId = s.id
           WHERE e.end IS NOT NULL"""
    return [(s, e, t) for s, e, t in db.execute(q) if t and t.startswith(prefix)]


def _intervals(db, table: str, cols: str = "start, end") -> list[tuple]:
    try:
        return sorted(db.execute(f"SELECT {cols} FROM {table}"))
    except sqlite3.OperationalError:
        return []


def _within(iv: list[tuple], s: int, e: int) -> list[tuple]:
    import bisect
    i = bisect.bisect_left(iv, (s,))
    out = []
    while i < len(iv) and iv[i][0] < e:
        out.append(iv[i])
        i += 1
    return out


def summarize(path: Path, sidecar: Path | None) -> None:
    db = sqlite3.connect(str(path))
    infers = sorted(_ranges(db, "infer "))
    engines = sorted(_ranges(db, "engine "))
    kernels = _intervals(db, "CUPTI_ACTIVITY_KIND_KERNEL")
    copies = _intervals(db, "CUPTI_ACTIVITY_KIND_MEMCPY", "start, end, bytes, copyKind")
    syncs = []
    try:
        syncs = sorted(db.execute(
            """SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r
               JOIN StringIds s ON r.nameId = s.id WHERE s.value LIKE 'cudaStreamSynchronize%'"""))
    except sqlite3.OperationalError:
        pass
    side = json.loads(sidecar.read_text()) if sidecar and sidecar.exists() else {}
    n = len(infers)
    if not n:
        raise SystemExit("no 'infer' NVTX ranges in this trace")

    def busy(iv, s, e):
        return sum(min(x[1], e) - max(x[0], s) for x in _within(iv, s, e)) / 1e6

    stage = defaultdict(lambda: defaultdict(float))
    mb = side.get("engine_mb", {})
    for s, e, text in engines:
        name = text.split(" ", 1)[1]
        key = name.rstrip("0123456789").rstrip("_") or name
        st = stage[key]
        st["calls"] += 1
        st["wall"] += (e - s) / 1e6
        st["kernel"] += busy(kernels, s, e)
        st["copy"] += busy(copies, s, e)
        st["mb"] += mb.get(name, 0.0)
    total = sum((e - s) / 1e6 for s, e, _ in infers)
    eng_wall = sum(v["wall"] for v in stage.values())
    k_all = sum(busy(kernels, s, e) for s, e, _ in infers)
    c_all = sum(busy(copies, s, e) for s, e, _ in infers)
    h2d = sum(b for s, e, b, k in copies if k == 1 and any(a <= s < z for a, z, _ in infers))
    d2h = sum(b for s, e, b, k in copies if k == 2 and any(a <= s < z for a, z, _ in infers))
    n_sync = sum(len(_within(syncs, s, e)) for s, e, _ in infers)

    loop = sorted(side.get("loop_ms") or [0.0])
    print(f"# {side.get('model', path.stem)}: {n} traced inferences, "
          f"p50 {loop[len(loop) // 2]} ms under nsys\n")
    print("| stage | calls/infer | wall ms | GPU kernels ms | copies ms | launch+sync+py ms "
          "| engine MB read/infer | weights GB/s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, v in stage.items():
        over = v["wall"] - v["kernel"] - v["copy"]
        gbps = (v["mb"] / 1e3) / (v["kernel"] / 1e3) if v["kernel"] and v["mb"] else None
        print(f"| {key} | {v['calls'] / n:g} | {v['wall'] / n:.1f} | {v['kernel'] / n:.1f} | "
              f"{v['copy'] / n:.1f} | {over / n:.1f} | {v['mb'] / n:.0f} | "
              f"{f'{gbps:.0f}' if gbps else '—'} |")
    print(f"| host glue between engines | | {(total - eng_wall) / n:.1f} | | | | | |")
    print(f"| **total** | | **{total / n:.1f}** | **{k_all / n:.1f}** | {c_all / n:.1f} | | | |")
    print(f"\nGPU kernels busy {k_all / total * 100:.0f} % of the wall time. Per inference: "
          f"{n_sync / n:.0f} stream synchronizations, {h2d / n / 1e6:.1f} MB host->device, "
          f"{d2h / n / 1e6:.1f} MB device->host.")
    top = defaultdict(lambda: [0.0, 0])
    try:
        for s, e, name in db.execute(
                """SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
                   JOIN StringIds s ON k.shortName = s.id"""):
            top[name][0] += (e - s) / 1e6
            top[name][1] += 1
    except sqlite3.OperationalError:
        pass
    if top:
        print("\n| kernel | ms/infer | launches/infer |\n|---|---:|---:|")
        for name, (ms, c) in sorted(top.items(), key=lambda x: -x[1][0])[:12]:
            print(f"| `{name[:70]}` | {ms / n:.1f} | {c / n:g} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", type=Path, help="an exported .sqlite to summarize")
    ap.add_argument("--sidecar", default=None,
                    help="trace-time sidecar JSON (default <model>.trace.json / beside the sqlite)")
    ap.add_argument("--model")
    ap.add_argument("--bundle")
    ap.add_argument("--task", default=None)
    ap.add_argument("--views", type=int, default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--chain", choices=["host", "device", "graph"], default="host",
                    help="host labels every engine call; device and graph show kernels only")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    if a.summarize:
        side = Path(a.sidecar) if a.sidecar else a.summarize.with_suffix(".trace.json")
        summarize(a.summarize, side)
    else:
        if not (a.model and a.bundle):
            ap.error("--model and --bundle are required to trace")
        trace(a)


if __name__ == "__main__":
    main()
