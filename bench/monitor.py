"""System-load sampling: RAM, CPU, GPU, temperature, power — Jetson first.

Two samplers, one interface. `TegrastatsMonitor` parses `tegrastats`, which is the
only source on a Jetson that reports **GPU utilisation and the board power rails** at
once. `PsutilMonitor` is the fallback for any other machine (the DGX Spark, a laptop)
so the harness can be developed and smoke-tested off the target board.

Why sample the whole system rather than just our process
--------------------------------------------------------
ORT/TensorRT can spawn worker threads and unified GPU allocations are not owned by
a separately measurable GPU process. A process-only number can therefore miss real
costs. So the primary metrics are system-wide deltas
measured against an idle baseline taken with the model already loaded:

    idle window   model resident, no inference   -> baseline
    load window   inference running flat out     -> measurement
    delta         what the inference itself costs

`procwatch.py` adds per-PID RSS/CPU on top, for attributing that delta.

Power
-----
Orin reports rails as `VDD_IN 4152mW/4152mW` (instantaneous/running-average).
`VDD_IN` is the whole board at the barrel jack — the number that matters for a
battery budget. `VDD_CPU_GPU_CV` and `VDD_SOC` split it. We integrate the
instantaneous value over the measurement window to get **energy per inference**
(mJ), which is the honest cross-backend comparison: 25 W for 100 ms beats 15 W for
500 ms, and a plain watt figure hides that.
"""

from __future__ import annotations

import re
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field

# tegrastats emits one line per interval; every field below is optional because the
# exact set differs between carrier boards and JetPack releases. Parse what is there.
_RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_RE_SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
_RE_CPU = re.compile(r"CPU \[([^\]]*)\]")
_RE_CPU_CORE = re.compile(r"(\d+)%@(\d+)")
_RE_GR3D = re.compile(r"GR3D_FREQ (\d+)%")
_RE_TEMP = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@([\d.]+)C")
_RE_POWER = re.compile(r"\b([A-Z][A-Z0-9_]+) (\d+)mW/(\d+)mW")


@dataclass
class Sample:
    t: float
    ram_used_mb: float | None = None
    ram_total_mb: float | None = None
    swap_used_mb: float | None = None
    cpu_pct_cores: list[float] = field(default_factory=list)
    gpu_pct: float | None = None
    temps_c: dict[str, float] = field(default_factory=dict)
    power_mw: dict[str, float] = field(default_factory=dict)

    @property
    def cpu_pct_total(self) -> float | None:
        """Sum across cores: 600 means six cores pinned on a 6-core Orin Nano."""
        return sum(self.cpu_pct_cores) if self.cpu_pct_cores else None


def _summ(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    return {
        "mean": round(statistics.fmean(s), 2),
        "p50": round(s[len(s) // 2], 2),
        "p95": round(s[min(len(s) - 1, int(len(s) * 0.95))], 2),
        "max": round(s[-1], 2),
        "min": round(s[0], 2),
    }


class _BaseMonitor:
    """Background sampler with named windows.

    Usage:
        mon.start()
        with mon.window("idle"): ...
        with mon.window("load"): ...
        mon.stop(); mon.summary()
    """

    interval_s = 0.1

    def __init__(self) -> None:
        self._samples: list[Sample] = []
        self._windows: dict[str, tuple[float, float]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- windows ------------------------------------------------------------
    class _Window:
        def __init__(self, mon: "_BaseMonitor", name: str):
            self.mon, self.name = mon, name

        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, *exc):
            self.mon._windows[self.name] = (self.t0, time.time())
            return False

    def window(self, name: str) -> "_BaseMonitor._Window":
        return self._Window(self, name)

    # -- reporting ----------------------------------------------------------
    def summary(self) -> dict:
        out: dict = {"sampler": type(self).__name__, "n_samples": len(self._samples),
                     "windows": {}}
        for name, (t0, t1) in self._windows.items():
            sel = [s for s in self._samples if t0 <= s.t <= t1]
            out["windows"][name] = self._summarize(sel, t1 - t0)
        # The comparison that answers "what does inference itself cost".
        if "idle" in out["windows"] and "load" in out["windows"]:
            out["delta_load_minus_idle"] = _delta(out["windows"]["idle"],
                                                  out["windows"]["load"])
        return out

    def _summarize(self, sel: list[Sample], duration_s: float) -> dict:
        if not sel:
            return {"duration_s": round(duration_s, 2), "n": 0}
        d: dict = {"duration_s": round(duration_s, 2), "n": len(sel)}
        ram = [s.ram_used_mb for s in sel if s.ram_used_mb is not None]
        if ram:
            d["ram_used_mb"] = _summ(ram)
            d["ram_total_mb"] = sel[-1].ram_total_mb
        swap = [s.swap_used_mb for s in sel if s.swap_used_mb is not None]
        if swap:
            d["swap_used_mb"] = _summ(swap)
        cpu = [s.cpu_pct_total for s in sel if s.cpu_pct_total is not None]
        if cpu:
            d["cpu_pct_total"] = _summ(cpu)          # 100 == one full core
            d["cpu_cores_busy"] = round(statistics.fmean(cpu) / 100.0, 2)
        gpu = [s.gpu_pct for s in sel if s.gpu_pct is not None]
        if gpu:
            d["gpu_pct"] = _summ(gpu)
        temps: dict[str, list[float]] = {}
        for s in sel:
            for k, v in s.temps_c.items():
                temps.setdefault(k, []).append(v)
        if temps:
            d["temp_c"] = {k: {"mean": round(statistics.fmean(v), 1),
                               "max": round(max(v), 1)} for k, v in temps.items()}
        rails: dict[str, list[float]] = {}
        for s in sel:
            for k, v in s.power_mw.items():
                rails.setdefault(k, []).append(v)
        if rails:
            d["power_mw"] = {k: _summ(v) for k, v in rails.items()}
            # Energy over the window, from the instantaneous rail readings.
            for k, v in rails.items():
                d.setdefault("energy_mj", {})[k] = round(
                    statistics.fmean(v) * duration_s, 1)
        return d

    # -- helpers ------------------------------------------------------------
    def _record(self, s: Sample) -> None:
        self._samples.append(s)


def _delta(idle: dict, load: dict) -> dict:
    """load - idle for the fields where a delta is meaningful."""
    out: dict = {}

    def sub(key: str, stat: str = "mean"):
        a, b = idle.get(key), load.get(key)
        if isinstance(a, dict) and isinstance(b, dict) and stat in a and stat in b:
            return round(b[stat] - a[stat], 2)
        return None

    for key in ("ram_used_mb", "cpu_pct_total", "gpu_pct"):
        v = sub(key)
        if v is not None:
            out[key] = v
    if "cpu_cores_busy" in idle and "cpu_cores_busy" in load:
        out["cpu_cores_busy"] = round(load["cpu_cores_busy"] - idle["cpu_cores_busy"], 2)
    for rail in set(idle.get("power_mw", {})) & set(load.get("power_mw", {})):
        out.setdefault("power_mw", {})[rail] = round(
            load["power_mw"][rail]["mean"] - idle["power_mw"][rail]["mean"], 1)
    return out


class TegrastatsMonitor(_BaseMonitor):
    """The Jetson sampler. Needs `tegrastats` on PATH (ships with L4T)."""

    def __init__(self, interval_ms: int = 100, binary: str = "tegrastats"):
        super().__init__()
        self.interval_ms = interval_ms
        self.binary = binary
        self.interval_s = interval_ms / 1000.0
        self._proc: subprocess.Popen | None = None

    @staticmethod
    def available() -> bool:
        return shutil.which("tegrastats") is not None

    def _run(self) -> None:
        self._proc = subprocess.Popen(
            [self.binary, "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            self._record(self.parse(line))
        try:
            self._proc.terminate()
        except Exception:
            pass

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        super().stop()

    @staticmethod
    def parse(line: str, now: float | None = None) -> Sample:
        s = Sample(t=now if now is not None else time.time())
        if (m := _RE_RAM.search(line)):
            s.ram_used_mb, s.ram_total_mb = float(m.group(1)), float(m.group(2))
        if (m := _RE_SWAP.search(line)):
            s.swap_used_mb = float(m.group(1))
        if (m := _RE_CPU.search(line)):
            # A parked core prints "off" rather than "N%@freq" — count it as 0.
            for part in m.group(1).split(","):
                core = _RE_CPU_CORE.search(part)
                s.cpu_pct_cores.append(float(core.group(1)) if core else 0.0)
        if (m := _RE_GR3D.search(line)):
            s.gpu_pct = float(m.group(1))
        s.temps_c = {k: float(v) for k, v in _RE_TEMP.findall(line)}
        # group(2) is instantaneous, group(3) tegrastats' own running average.
        s.power_mw = {k: float(cur) for k, cur, _avg in _RE_POWER.findall(line)}
        return s


class PsutilMonitor(_BaseMonitor):
    """Off-Jetson fallback: psutil for CPU/RAM, nvidia-smi for GPU/power.

    Not a substitute for tegrastats on the target — there is no unified-memory
    accounting and no board power rail. It exists so the harness can be exercised
    end to end on a development machine before it ever touches the Orin.
    """

    def __init__(self, interval_s: float = 0.5):
        super().__init__()
        self.interval_s = interval_s
        self._smi = shutil.which("nvidia-smi")
        self._n = 0

    def _run(self) -> None:
        import psutil

        psutil.cpu_percent(percpu=True)  # prime the first delta
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            vm = psutil.virtual_memory()
            s = Sample(t=time.time(),
                       ram_used_mb=(vm.total - vm.available) / 1e6,
                       ram_total_mb=vm.total / 1e6,
                       swap_used_mb=psutil.swap_memory().used / 1e6,
                       cpu_pct_cores=list(psutil.cpu_percent(percpu=True)))
            # nvidia-smi costs ~100 ms of CPU per call; poll it every other
            # sample so the sampler does not distort what it is measuring.
            self._n += 1
            if self._smi and self._n % 2 == 0:
                s.gpu_pct, pw = self._nvidia_smi()
                if pw is not None:
                    s.power_mw["GPU_SMI"] = pw
            self._record(s)

    def _nvidia_smi(self) -> tuple[float | None, float | None]:
        try:
            out = subprocess.run(
                [self._smi, "--query-gpu=utilization.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2).stdout.strip().splitlines()
            util, power = (x.strip() for x in out[0].split(","))
            return (float(util) if util not in ("[N/A]", "N/A") else None,
                    float(power) * 1000 if power not in ("[N/A]", "N/A") else None)
        except Exception:
            return None, None


def make_monitor(force: str | None = None) -> _BaseMonitor:
    """tegrastats on a Jetson, psutil everywhere else. `force` overrides."""
    if force == "psutil":
        return PsutilMonitor()
    if force == "tegrastats" or (force is None and TegrastatsMonitor.available()):
        return TegrastatsMonitor()
    return PsutilMonitor()
