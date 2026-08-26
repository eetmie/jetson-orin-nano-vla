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

Drift
-----
Latency is reported per quartile as well as in aggregate, because a short run cannot
show whether latency holds. Whether this board sustains a VLA load at MAXN, or gives
clock back as it heats, is an open question here — `--duration-s` is how it gets asked,
and the quartile means are the answer. If quartile 4 is materially slower than quartile
1, that is the figure a deployment lives with rather than the mean.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import subprocess
import tempfile
import time
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
    # Linear interpolation is the conventional percentile definition and behaves
    # sensibly for short smoke runs too.
    return float(np.percentile(np.asarray(sorted_vals, dtype=np.float64), q * 100.0))


def latency_stats(ms: list[float]) -> dict:
    if not ms:
        raise ValueError("cannot summarize an empty latency series")
    s = sorted(ms)
    n = len(s)
    # Assign every sample to one of the four ordered windows. The old n//4 slicing
    # silently dropped the last n%4 calls and could hide end-of-run drift.
    quartiles = [round(float(np.mean(part)), 2)
                 for part in np.array_split(np.asarray(ms, dtype=np.float64), 4)
                 if len(part)]
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


def _array_digest(h, value: np.ndarray) -> None:
    """Hash an array without making identity depend on Python object layout."""
    a = np.ascontiguousarray(value)
    h.update(a.dtype.str.encode())
    h.update(str(tuple(a.shape)).encode())
    h.update(a.tobytes())


def observation_digests(observation) -> tuple[str, str]:
    """Exact raw-observation and injected-noise identities used by parity."""
    h = hashlib.sha256()
    h.update(str(observation.index).encode())
    h.update(observation.task.encode())
    for image in observation.images:
        _array_digest(h, image)
    _array_digest(h, observation.state)
    _array_digest(h, observation.noise)

    noise = hashlib.sha256()
    _array_digest(noise, observation.noise)
    return h.hexdigest(), noise.hexdigest()


def _validated_infer(result, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Enforce the model-execution part of the result contract."""
    chunk = np.asarray(result.chunk)
    if chunk.ndim != 2 or not all(int(x) > 0 for x in chunk.shape):
        raise ValueError(f"action chunk must be a non-empty 2-D array, got {chunk.shape}")
    if expected_shape is not None and chunk.shape != expected_shape:
        raise ValueError(
            f"action chunk shape changed from {expected_shape} to {chunk.shape}")
    if not np.all(np.isfinite(chunk)):
        raise ValueError("action chunk contains NaN or infinity")

    timings = result.timings_ms
    if not isinstance(timings, dict) or "total" not in timings:
        raise ValueError("backend timings must contain total")
    for key, value in timings.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"backend timing {key!r} is not finite: {value!r}")
        if key == "total" and value <= 0:
            raise ValueError(f"backend total timing must be positive, got {value!r}")
    return chunk


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
    env["jetson_clocks"] = (
        sh("sudo", "-n", "jetson_clocks", "--show")
        or sh("jetson_clocks", "--show"))
    clock_state = {}
    for name, pattern in (
        ("cpu_min_khz", "/sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq"),
        ("cpu_cur_khz", "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"),
        ("cpu_max_khz", "/sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq"),
        ("gpu_min_hz", "/sys/class/devfreq/*gpu*/min_freq"),
        ("gpu_cur_hz", "/sys/class/devfreq/*gpu*/cur_freq"),
        ("gpu_max_hz", "/sys/class/devfreq/*gpu*/max_freq"),
    ):
        import glob
        values = []
        for filename in sorted(glob.glob(pattern)):
            try:
                values.append(int(Path(filename).read_text().strip()))
            except (OSError, ValueError):
                pass
        if values:
            clock_state[name] = values
    env["clock_state"] = clock_state
    env["cuda"] = sh("bash", "-c", "nvcc --version 2>/dev/null | tail -2")
    for mod in ("torch", "onnxruntime", "tensorrt", "transformers", "lerobot", "numpy"):
        try:
            m = __import__(mod)
            env.setdefault("packages", {})[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    env["git_sha"] = sh("git", "rev-parse", "HEAD")
    env["git_dirty"] = bool(sh("git", "status", "--porcelain"))
    if env["git_dirty"]:
        try:
            diff = subprocess.run(
                ["git", "diff", "--binary", "HEAD"], capture_output=True,
                check=True, timeout=10).stdout
            env["git_diff_sha256"] = hashlib.sha256(diff).hexdigest()
        except Exception:
            env["git_diff_sha256"] = None
    return env


def run_benchmark(backend: Backend, obs: ObsSource, *, iters: int = 100,
                  warmup: int = 5, idle_s: float = 5.0, duration_s: float | None = None,
                  save_chunks: int = 8, fps: int = 30, monitor_kind: str | None = None,
                  label: str | None = None, notes: str | None = None,
                  obs_ring_size: int = 16) -> dict:
    result: dict = {
        "label": label or backend.name,
        "backend": backend.name,
        "status": "running",
        "notes": notes,
        "env": collect_env(),
        "obs": obs.describe(),
        "config": {"iters": iters, "warmup": warmup, "idle_s": idle_s,
                   "duration_s": duration_s, "fps": fps,
                   "obs_ring_size": obs_ring_size},
        "validity": {"execution": "not_run", "instrumentation": "not_checked",
                     "placement": "not_checked", "parity": "not_checked"},
    }

    if iters <= 0:
        raise ValueError("iters must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if idle_s < 0:
        raise ValueError("idle_s must be non-negative")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive when set")
    if save_chunks < 0:
        raise ValueError("save_chunks must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if obs_ring_size <= 0:
        raise ValueError("obs_ring_size must be positive")

    mon = None
    watch = None

    try:
        # Build a bounded deterministic ring before first inference and monitoring.
        # Duration runs wrap around it, so CPU/power/throughput describe runtime-only
        # work rather than synthetic generation or lazy frame decoding.
        ring_size = max(obs_ring_size, warmup + 1 + save_chunks)
        ring = []
        prep_ms = []
        input_sha256 = []
        noise_sha256 = []
        for index in range(ring_size):
            t_obs = time.perf_counter()
            observation = obs[index]
            prep_ms.append((time.perf_counter() - t_obs) * 1000.0)
            raw_hash, noise_hash = observation_digests(observation)
            ring.append(observation)
            input_sha256.append(raw_hash)
            noise_sha256.append(noise_hash)
        result["observation_prepare_ms"] = latency_stats(prep_ms)
        result["observation_ring"] = {
            "size": ring_size,
            "source_indices": [o.index for o in ring],
            "scope": "materialized before monitoring; measurement wraps this ring",
        }

        mon = make_monitor(monitor_kind)
        mon.start()
        time.sleep(1.0)

        t0 = time.perf_counter()
        backend.load()
        result["load_s"] = round(time.perf_counter() - t0, 2)
        result["meta"] = backend.meta()

        watch = ProcWatch(backend.pids())
        watch.start()
        result["rss_after_load_mb"] = round(watch.rss_now_mb(), 1)

        t = time.perf_counter()
        first = backend.infer(ring[0])
        result["first_infer_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        first_chunk = _validated_infer(first)
        expected_shape = tuple(first_chunk.shape)
        result["chunk_shape"] = list(expected_shape)
        # Some backends learn metadata during the first inference.
        result["meta"] = backend.meta()

        for i in range(warmup):
            _validated_infer(backend.infer(ring[i + 1]), expected_shape)

        i0 = time.time()
        with mon.window("idle"):
            time.sleep(idle_s)
        watch.mark("idle", i0, time.time())

        lat: list[float] = []
        infer_wall: list[float] = []
        observation_ms: list[float] = []
        detail: dict[str, list[float]] = {}
        chunks: list[list] = []
        saved_indices: list[int] = []
        saved_input_hashes: list[str] = []
        saved_noise_hashes: list[str] = []
        l0 = time.time()
        with mon.window("load"):
            start = time.perf_counter()
            i = 0
            while True:
                ring_pos = (warmup + 1 + i) % ring_size
                t_obs = time.perf_counter()
                observation = ring[ring_pos]
                observation_ms.append((time.perf_counter() - t_obs) * 1000.0)

                t_infer = time.perf_counter()
                infer_result = backend.infer(observation)
                infer_wall.append((time.perf_counter() - t_infer) * 1000.0)
                chunk = _validated_infer(infer_result, expected_shape)
                lat.append(infer_result.timings_ms["total"])
                for key, value in infer_result.timings_ms.items():
                    if key != "total" and isinstance(value, (int, float)):
                        detail.setdefault(key, []).append(float(value))
                if i < save_chunks:
                    chunks.append(np.asarray(chunk, dtype=np.float32).tolist())
                    saved_indices.append(observation.index)
                    saved_input_hashes.append(input_sha256[ring_pos])
                    saved_noise_hashes.append(noise_sha256[ring_pos])
                i += 1
                if duration_s is not None:
                    if time.perf_counter() - start >= duration_s:
                        break
                elif i >= iters:
                    break
            measured_window_s = time.perf_counter() - start
        watch.mark("load", l0, time.time())

        result["latency_ms"] = latency_stats(lat)
        result["infer_wall_ms"] = latency_stats(infer_wall)
        result["observation_ms"] = latency_stats(observation_ms)
        result["measurement"] = {
            "scope": "runtime-only over a pre-materialized observation ring",
            "completed_calls": len(lat),
            "window_s": round(measured_window_s, 6),
            "achieved_hz": round(len(lat) / measured_window_s, 4),
        }
        result["latency_breakdown_ms"] = {
            key: round(statistics.fmean(values), 3)
            for key, values in sorted(detail.items())
        }
        result["saved_chunks"] = {
            "noise_injected": backend.noise_injected,
            "obs_indices": saved_indices,
            "seed": obs.seed,
            "input_sha256": saved_input_hashes,
            "noise_sha256": saved_noise_hashes,
            "chunks": chunks,
        }

        chunk_size = result["meta"].get("chunk_size") or expected_shape[0]
        cycle_mean_ms = measured_window_s / len(lat) * 1000.0
        consumed = cycle_mean_ms / 1000.0 * fps
        result["control"] = {
            "fps": fps,
            "chunk_size": chunk_size,
            "steps_consumed_per_inference_mean": round(consumed, 2),
            "steps_consumed_per_inference_p95": round(
                result["infer_wall_ms"]["p95"] / 1000.0 * fps, 2),
            "chunk_headroom_x": round(chunk_size / consumed, 2) if consumed else None,
            "note": "headroom < 1 means the plan runs out before the next one lands",
        }
        result["meta"] = backend.meta()
        result["validity"]["execution"] = "pass"
        result["status"] = "ok"

    except Exception as e:
        import traceback
        result["status"] = "failed"
        result["validity"]["execution"] = "fail"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    finally:
        if mon is not None:
            mon.stop()
        try:
            backend.close()
        except Exception:
            pass
        if watch is not None:
            watch.stop()
            result["process"] = watch.summary()
        if mon is not None:
            result["system"] = mon.summary()

        system_windows = result.get("system", {}).get("windows", {})
        process_windows = result.get("process", {}).get("windows", {})
        instrumented = all(
            (windows.get(name) or {}).get("n", 0) > 0
            for windows in (system_windows, process_windows)
            for name in ("idle", "load")
        )
        result["validity"]["instrumentation"] = "pass" if instrumented else "fail"

    return result


def write_result(result: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, allow_nan=False)
    # A reader sees either the previous complete result or the new complete result.
    with tempfile.NamedTemporaryFile("w", dir=out.parent, prefix=f".{out.name}.",
                                     delete=False) as f:
        f.write(payload)
        f.flush()
        tmp = Path(f.name)
    tmp.replace(out)
    return out
