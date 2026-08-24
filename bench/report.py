"""Turn a directory of run JSONs into the markdown tables that go in RESULTS.md."""

from __future__ import annotations

import json
from pathlib import Path

from .parity import load_results, parity_report


def _g(d: dict, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _w(mw):
    return round(mw / 1000.0, 2) if isinstance(mw, (int, float)) else None


def _md_table(rows: list[list], header: list[str]) -> str:
    def cell(x):
        return "—" if x is None else str(x)
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(out)


def speed_table(runs: list[dict]) -> str:
    rows = []
    for r in runs:
        if r.get("status") != "ok":
            rows.append([r.get("label"), "**FAILED**", *["—"] * 7])
            continue
        lat = r.get("latency_ms", {})
        rows.append([
            r.get("label"), "ok", lat.get("p50"), lat.get("p95"), lat.get("max"),
            lat.get("hz_mean"), r.get("first_infer_ms"), r.get("load_s"),
            lat.get("drift_q4_vs_q1_pct"),
        ])
    return _md_table(rows, [
        "run", "status", "p50 ms", "p95 ms", "max ms", "Hz", "1st infer ms",
        "load s", "drift q4/q1 %"])


def footprint_table(runs: list[dict]) -> str:
    rows = []
    for r in runs:
        if r.get("status") != "ok":
            continue
        sysm = r.get("system", {})
        load_w = _g(sysm, "windows", "load", default={})
        delta = sysm.get("delta_load_minus_idle", {})
        proc_load = _g(r, "process", "windows", "load", default={})
        proc_idle = _g(r, "process", "windows", "idle", default={})
        energy = _g(load_w, "energy_mj", "VDD_IN")
        n = (r.get("latency_ms") or {}).get("n") or 1
        rows.append([
            r.get("label"),
            r.get("rss_after_load_mb"),
            _g(proc_load, "rss_mb", "max"),
            _g(load_w, "ram_used_mb", "mean"),
            delta.get("ram_used_mb"),
            _g(load_w, "gpu_pct", "mean"),
            _g(proc_idle, "cores_busy"),
            proc_load.get("cores_busy"),
            _w(_g(load_w, "power_mw", "VDD_IN", "mean")),
            round(energy / n, 1) if energy else None,
            _g(load_w, "temp_c", "tj", "max"),
        ])
    return _md_table(rows, [
        "run", "RSS loaded MB", "RSS peak MB", "sys RAM MB", "ΔRAM MB", "GPU %",
        "CPU cores idle", "CPU cores busy", "VDD_IN W", "mJ/infer", "tj max °C"])


def breakdown_table(runs: list[dict]) -> str:
    keys: list[str] = []
    for r in runs:
        for k in (r.get("latency_breakdown_ms") or {}):
            if k not in keys and not k.endswith(".calls"):
                keys.append(k)
    if not keys:
        return "_(no per-stage timings recorded)_"
    rows = []
    for r in runs:
        b = r.get("latency_breakdown_ms") or {}
        if not b:
            continue
        rows.append([r.get("label")] + [b.get(k) for k in keys])
    return _md_table(rows, ["run"] + [f"{k} ms" for k in keys])


def control_table(runs: list[dict]) -> str:
    rows = []
    for r in runs:
        c = r.get("control")
        if not c:
            continue
        rows.append([r.get("label"), c.get("fps"), c.get("chunk_size"),
                     c.get("steps_consumed_per_inference_mean"),
                     c.get("steps_consumed_per_inference_p95"),
                     c.get("chunk_headroom_x")])
    if not rows:
        return "_(no control-loop figures recorded)_"
    return _md_table(rows, ["run", "fps", "chunk", "steps used (mean)",
                            "steps used (p95)", "headroom ×"])


def parity_table(runs: list[dict], prefer_ref: str | None = None) -> str:
    rep = parity_report(runs, prefer_ref)
    if "error" in rep:
        return f"_{rep['error']}_"
    rows = []
    for c in rep["comparisons"]:
        rows.append([c.get("candidate"), c.get("verdict"), c.get("mode"),
                     c.get("cosine_min"), c.get("max_abs_diff"),
                     c.get("max_abs_diff_pct_of_range"),
                     c.get("max_dim_mean_shift")])
    head = f"Reference: **{rep['reference']}**\n\n"
    return head + _md_table(rows, [
        "candidate", "verdict", "mode", "cosine min", "max abs diff",
        "% of range", "max dim mean shift"])


def env_table(runs: list[dict]) -> str:
    rows = []
    for r in runs:
        e = r.get("env", {})
        pkg = e.get("packages", {})
        rows.append([r.get("label"), e.get("hostname"), e.get("l4t") or e.get("kernel"),
                     pkg.get("torch"), pkg.get("onnxruntime"), pkg.get("tensorrt"),
                     e.get("git_sha")])
    return _md_table(rows, ["run", "host", "L4T / kernel", "torch", "onnxruntime",
                            "tensorrt", "repo sha"])


def build_report(paths: list[Path], prefer_ref: str | None = None) -> str:
    runs = load_results(paths)
    runs.sort(key=lambda r: (r.get("status") != "ok", r.get("label") or ""))
    if not runs:
        return "_no result JSONs found_"

    parts = [
        "# Results",
        "",
        "Generated by `python -m bench report`. Every row is one run JSON in "
        "`results/`; the JSON holds far more than fits here.",
        "",
        "## Speed",
        "",
        speed_table(runs),
        "",
        "`1st infer` is the very first call after load — a lazy TensorRT build, a "
        "cuDNN autotune or a CUDA context lands there. `drift` compares the last "
        "quarter of the run against the first: a positive number is the board "
        "throttling under sustained load, which is what a real deployment lives with.",
        "",
        "## Footprint",
        "",
        footprint_table(runs),
        "",
        "`CPU cores busy` is per-process CPU during the measurement window — 1.0 means "
        "one of the six Orin Nano cores is gone and the robot control stack cannot have "
        "it. `CPU cores idle` is the same measure with the model loaded but not "
        "inferring, so the difference is what inference itself takes. `ΔRAM` is "
        "system-wide load minus idle: the cost of *running*, on top of the resident "
        "weights. `mJ/infer` integrates VDD_IN (whole board) over the window and "
        "divides by the inference count — the fair way to compare a fast-and-hungry "
        "backend against a slow-and-frugal one.",
        "",
        "## Where the time goes",
        "",
        breakdown_table(runs),
        "",
        "## What it means for the control loop",
        "",
        control_table(runs),
        "",
        "SmolVLA emits a chunk of actions authored at the dataset fps. The controller "
        "plays them while the next inference runs, so `steps used` is how much of the "
        "chunk is consumed per replan and `headroom ×` is how much plan is left over. "
        "Below 1× the plan runs dry before the next one lands.",
        "",
        "## Parity",
        "",
        parity_table(runs, prefer_ref),
        "",
        "## Environment",
        "",
        env_table(runs),
        "",
    ]
    failed = [r for r in runs if r.get("status") != "ok"]
    if failed:
        parts += ["## Failures", ""]
        for r in failed:
            parts += [f"### {r.get('label')}", "", "```",
                      str(r.get("error")), "```", ""]
    return "\n".join(parts)
