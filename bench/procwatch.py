"""Per-process RSS and CPU, sampled from /proc — no psutil required on the target.

The system-wide monitor answers *how much* the machine is doing; this answers *who*.
It matters for one question in particular: the split-ONNX backend runs its text
encoder, five projectors and the whole flow-matching denoise loop as numpy on the
CPU, so "GPU-only" is a claim to be measured rather than assumed. A PID list is
accepted so launchers and recursively spawned worker processes can be included.

CPU percentages come from utime+stime deltas over wall time: 100 == one core fully
busy, 600 == all six Orin Nano cores.
"""

from __future__ import annotations

import os
import statistics
import threading
import time
from pathlib import Path

_CLK_TCK = os.sysconf("SC_CLK_TCK")
_PAGE_KB = os.sysconf("SC_PAGE_SIZE") / 1024


def _read_stat(pid: int) -> tuple[float, float] | None:
    """(cpu_seconds, rss_mb) for a pid, or None if it is gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # comm may contain spaces and parentheses; fields are positional after ')'.
        fields = stat[stat.rindex(")") + 2:].split()
        utime, stime = float(fields[11]), float(fields[12])
        rss_pages = float(fields[21])
        return (utime + stime) / _CLK_TCK, rss_pages * _PAGE_KB / 1024
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def children_of(pid: int) -> list[int]:
    """Direct children; ProcWatch walks this recursively."""
    try:
        kids = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        return [int(k) for k in kids]
    except Exception:
        return []


class ProcWatch:
    """Sample a set of PIDs (and their children) in the background."""

    def __init__(self, pids: list[int] | None = None, interval_s: float = 0.25,
                 follow_children: bool = True):
        self.pids = list(pids or [os.getpid()])
        self.interval_s = interval_s
        self.follow_children = follow_children
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._series: list[tuple[float, float, float]] = []  # (t, cpu_pct, rss_mb)
        self._windows: dict[str, tuple[float, float]] = {}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def mark(self, name: str, t0: float, t1: float) -> None:
        self._windows[name] = (t0, t1)

    def _run(self) -> None:
        prev_cpu, prev_t = self._snapshot()
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            cpu, rss = self._snapshot_full()
            now = time.time()
            dt = now - prev_t
            if dt > 0:
                self._series.append((now, (cpu - prev_cpu) / dt * 100.0, rss))
            prev_cpu, prev_t = cpu, now

    def _all_pids(self) -> list[int]:
        pids = list(dict.fromkeys(self.pids))
        if self.follow_children:
            # Walk the whole process tree. A launcher -> server -> worker chain is common
            # and direct children alone silently omitted the process doing inference.
            seen = set(pids)
            pending = list(pids)
            while pending:
                for child in children_of(pending.pop()):
                    if child not in seen:
                        seen.add(child)
                        pids.append(child)
                        pending.append(child)
        return pids

    def _snapshot(self) -> tuple[float, float]:
        cpu, _ = self._snapshot_full()
        return cpu, time.time()

    def _snapshot_full(self) -> tuple[float, float]:
        cpu = rss = 0.0
        for pid in self._all_pids():
            if (r := _read_stat(pid)):
                cpu += r[0]
                rss += r[1]
        return cpu, rss

    def rss_now_mb(self) -> float:
        return self._snapshot_full()[1]

    def summary(self) -> dict:
        out: dict = {"pids": self.pids, "n_samples": len(self._series), "windows": {}}
        for name, (t0, t1) in self._windows.items():
            sel = [(c, r) for t, c, r in self._series if t0 <= t <= t1]
            if not sel:
                out["windows"][name] = {"n": 0}
                continue
            cpu = sorted(c for c, _ in sel)
            rss = sorted(r for _, r in sel)
            out["windows"][name] = {
                "n": len(sel),
                # 100 == one core. Divide by 100 for "cores this backend takes away
                # from the robot control stack".
                "cpu_pct": {"mean": round(statistics.fmean(cpu), 1),
                            "p95": round(cpu[min(len(cpu) - 1, int(len(cpu) * .95))], 1),
                            "max": round(cpu[-1], 1)},
                "cores_busy": round(statistics.fmean(cpu) / 100.0, 2),
                "rss_mb": {"mean": round(statistics.fmean(rss), 1),
                           "max": round(rss[-1], 1)},
            }
        return out
