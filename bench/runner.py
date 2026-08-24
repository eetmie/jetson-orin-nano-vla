"""The measurement loop: warm up, sit idle, run flat out, write one JSON per run.

Shape of a run
--------------
    load          construct the backend and load weights/engines  -> load_s
    first infer   the very first call, kept separate              -> first_infer_ms
    warmup        N calls, discarded                              (caches, autotune)
    idle window   model resident, nothing running                 -> the baseline
    measure       M calls back to back, monitored                 -> the numbers

The idle window is what makes the RAM/CPU/power figures mean something. A backend's
resident cost (weights, engines, arenas) and its *running* cost are different
questions, and on an 8 GB unified-memory board with a robot control stack to host,
both matter. Reporting one number for "RAM" would answer neither.

Thermal drift
-------------
Latency is reported per quartile as well as in aggregate. An Orin Nano at MAXN with
pinned clocks will sustain for a while and then thermal-throttle; a benchmark that
runs for ten seconds and quotes a mean can miss it entirely. If quartile 4 is
materially slower than quartile 1, that is the number a real deployment lives with.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .backends.base import Backend
from .monitor import make_monitor
from .obs import ObsSource
from .procwatch import ProcWatch


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]


def latency_stats(ms: list[float]) -> dict:
    s = sorted(ms)
    n = len(s)
    q = max(1, n // 4)
    quartiles = [round(statistics.fmean(ms[i * q:(i + 1) * q]), 2) for i in range(4)
                 if ms[i * q:(i + 1) * q]]
    out = {
        "n": n,
        "mean": round(statistics.fmean(s), 2),
        "p50": round(_pct(s, 0.50), 2),
        "p90": round(_pct(s, 0.90), 2),
        "p95": round(_pct(s, 0.95), 2),
        "p99": round(_pct(s, 0.99), 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "std": round(statistics.pstdev(s), 2) if n > 1 else 0.0,
        "hz_mean": round(1000.0 / statistics.fmean(s), 2),
        # In arrival order, not sorted — this is the drift check.
        "quartile_means_in_order": quartiles,
    }
    if len(quartiles) == 4 and quartiles[0] > 0:
        out["drift_q4_vs_q1_pct"] = round(
            (quartiles[3] - quartiles[0]) / quartiles[0] * 100.0, 1)
    return out


def collect_env() -> dict:
    """A fingerprint of the machine, so a number can be traced to a state."""
    def sh(*cmd, timeout=10):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip() or None
        except Exception:
            return None

    env: dict = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for path, key in (("/etc/nv_tegra_release", "l4t"),
                      ("/etc/os-release", "os_release")):
        p = Path(path)
        if p.exists():
            env[key] = p.read_text().strip().splitlines()[0]
    # Power mode and clock pinning change every number in this repo, so record them.
    env["nvpmodel"] = sh("nvpmodel", "-q")
    env["jetson_clocks"] = sh("bash", "-c", "jetson_clocks --show 2>/dev/null | head -4")
    env["cuda"] = sh("bash", "-c", "nvcc --version 2>/dev/null | tail -2")
    for mod in ("torch", "onnxruntime", "tensorrt", "transformers", "lerobot", "numpy"):
        try:
            m = __import__(mod)
            env.setdefault("packages", {})[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    env["git_sha"] = sh("git", "rev-parse", "--short", "HEAD")
    env["git_dirty"] = bool(sh("git", "status", "--porcelain"))
    return env


def run_benchmark(backend: Backend, obs: ObsSource, *, iters: int = 100,
                  warmup: int = 5, idle_s: float = 5.0, duration_s: float | None = None,
                  save_chunks: int = 8, fps: int = 30, monitor_kind: str | None = None,
                  label: str | None = None, notes: str | None = None) -> dict:
    result: dict = {
        "label": label or backend.name,
        "backend": backend.name,
        "status": "running",
        "notes": notes,
        "env": collect_env(),
        "obs": obs.describe(),
        "config": {"iters": iters, "warmup": warmup, "idle_s": idle_s,
                   "duration_s": duration_s, "fps": fps},
    }

    mon = make_monitor(monitor_kind)
    mon.start()
    time.sleep(1.0)                       # let the sampler get a few points down

    try:
        t0 = time.perf_counter()
        backend.load()
        result["load_s"] = round(time.perf_counter() - t0, 2)
        result["meta"] = backend.meta()

        watch = ProcWatch(backend.pids())
        watch.start()

        # Resident footprint with the model loaded but nothing running.
        result["rss_after_load_mb"] = round(watch.rss_now_mb(), 1)

        # The first call is where a lazy TRT build, a cuDNN autotune or a CUDA
        # context creation lands. Never fold it into the steady-state number, but
        # never hide it either: on a board that clears /tmp at boot it is the
        # difference between a 5-second start and a 5-minute one.
        t = time.perf_counter()
        first = backend.infer(obs[0])
        result["first_infer_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        result["chunk_shape"] = list(np.asarray(first.chunk).shape)

        for i in range(warmup):
            backend.infer(obs[i + 1])

        i0 = time.time()
        with mon.window("idle"):
            time.sleep(idle_s)
        watch.mark("idle", i0, time.time())

        lat: list[float] = []
        detail: dict[str, list[float]] = {}
        chunks: list[list] = []
        l0 = time.time()
        with mon.window("load"):
            start = time.perf_counter()
            i = 0
            while True:
                r = backend.infer(obs[warmup + 1 + i])
                lat.append(r.timings_ms["total"])
                for k, v in r.timings_ms.items():
                    if k != "total" and isinstance(v, (int, float)):
                        detail.setdefault(k, []).append(float(v))
                if i < save_chunks:
                    chunks.append(np.asarray(r.chunk, dtype=np.float32).tolist())
                i += 1
                if duration_s is not None:
                    if time.perf_counter() - start >= duration_s:
                        break
                elif i >= iters:
                    break
        watch.mark("load", l0, time.time())

        result["latency_ms"] = latency_stats(lat)
        result["latency_breakdown_ms"] = {
            k: round(statistics.fmean(v), 3) for k, v in sorted(detail.items())}
        result["saved_chunks"] = {
            "noise_injected": backend.noise_injected,
            "obs_indices": list(range(warmup + 1, warmup + 1 + len(chunks))),
            "seed": obs.seed,
            "chunks": chunks,
        }

        # Latency in the units the machine actually cares about: SmolVLA emits a
        # chunk of actions authored at the dataset fps, and the controller consumes
        # roughly `infer_s * fps` of them before the next plan lands. A chunk that
        # runs out before the replan is what makes the valves fall back to hold/decay.
        chunk_size = (result["meta"].get("chunk_size")
                      or np.asarray(first.chunk).shape[0])
        consumed = result["latency_ms"]["mean"] / 1000.0 * fps
        result["control"] = {
            "fps": fps,
            "chunk_size": chunk_size,
            "steps_consumed_per_inference_mean": round(consumed, 2),
            "steps_consumed_per_inference_p95": round(
                result["latency_ms"]["p95"] / 1000.0 * fps, 2),
            "chunk_headroom_x": round(chunk_size / consumed, 2) if consumed else None,
            "note": "headroom < 1 means the plan runs out before the next one lands",
        }
        result["status"] = "ok"

    except Exception as e:  # a failure IS a result — record it and move on
        import traceback
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    finally:
        mon.stop()
        try:
            backend.close()
        except Exception:
            pass
        w = locals().get("watch")
        if w is not None:
            w.stop()
            result["process"] = w.summary()
        result["system"] = mon.summary()

    return result


def write_result(result: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return out
