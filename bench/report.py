"""Turn a directory of run JSONs into the markdown tables that go in RESULTS.md."""

from __future__ import annotations

from pathlib import Path

from .parity import load_results


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


def _cams(r: dict) -> str:
    """Real cameras fed, over the slots the export declares -- e.g. "2/2", "1/3".

    Never compare two rows whose `cams` differ and call it a speedup: a padded slot
    costs no vision pass but still occupies its prefix tokens, so fewer real cameras is
    a different OBSERVATION, not the same work done faster. Parity across differing
    cams is meaningless for the same reason -- it reports a cosine against a sequence
    the reference never saw.
    """
    views = _g(r, "model", "views", default=None)
    # Each family names the declared count differently: smolvla carries cam_slots (the
    # slots its published export was built with), xvla carries num_views (the
    # checkpoint's num_image_views). Same concept, so the column reads the same.
    slots = (_g(r, "model", "cam_slots", default=None)
             or _g(r, "meta", "n_cam_slots", default=None)
             or _g(r, "meta", "num_views", default=None))
    if views is None:
        return "—"
    return f"{views}/{slots}" if slots else str(views)


def _configured_eps(r: dict) -> str:
    """Configured first-choice provider per graph, not measured node placement."""
    per = (_g(r, "meta", "configured_provider_priority_per_graph", default=None)
           or _g(r, "meta", "providers_per_graph", default=None))
    if not per:
        return "—"
    from collections import Counter
    counts = Counter(v.replace("ExecutionProvider", "") for v in per.values())
    parts = [f"{counts[key]} {key.replace('Tensorrt', 'TRT')}"
             for key in ("Tensorrt", "CUDA", "CPU") if counts.get(key)]
    suffix = " (legacy)" if "configured_provider_priority_per_graph" not in (
        r.get("meta") or {}) else ""
    return " / ".join(parts) + suffix


def speed_table(runs: list[dict]) -> str:
    rows = []
    for r in runs:
        if r.get("status") != "ok":
            rows.append([r.get("label"), "**FAILED**", *["—"] * 9])
            continue
        lat = r.get("latency_ms", {})
        outer = r.get("infer_wall_ms") or lat
        achieved = _g(r, "measurement", "achieved_hz", default=lat.get("hz_mean"))
        rows.append([
            r.get("label"), "ok", _cams(r), lat.get("p50"), outer.get("p50"),
            outer.get("p95"), achieved, _configured_eps(r), r.get("first_infer_ms"),
            r.get("load_s"), outer.get("drift_q4_vs_q1_pct"),
        ])
    return _md_table(rows, [
        "run", "status", "cams", "backend p50 ms", "outer p50 ms", "outer p95 ms",
        "achieved Hz", "configured EP priority", "1st infer ms", "load s",
        "drift q4/q1 %"])


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
    headers = [key if key.endswith(".steps") else f"{key} ms" for key in keys]
    return _md_table(rows, ["run"] + headers)


def control_summary(runs: list[dict]) -> str:
    entries = []
    rates = set()
    for r in runs:
        c = r.get("control")
        if not c or r.get("status") != "ok":
            continue
        fps = c.get("fps")
        chunk = c.get("chunk_size")
        consumed = c.get("steps_consumed_per_inference_mean")
        if not all(isinstance(v, (int, float)) for v in (fps, chunk, consumed)):
            continue
        rates.add(float(fps))
        remaining = float(chunk) - float(consumed)
        margin = (f"{remaining:.1f} steps remain" if remaining >= 0 else
                  f"it overruns by {-remaining:.1f} steps")
        entries.append(
            f"`{r.get('label')}` spans {float(consumed):.1f} of its "
            f"{float(chunk):g}-step action chunk ({margin})")
    if not entries:
        return "_(no control-loop figures recorded)_"
    rate = next(iter(rates)) if len(rates) == 1 else None
    prefix = f"At a {rate:g} Hz control rate, " if rate is not None else ""
    return prefix + "; ".join(entries) + "."


def build_report(paths: list[Path]) -> str:
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
        "`backend p50` is the backend-reported total; `outer p50` wraps the complete "
        "backend call; `achieved Hz` is completed calls divided by the measured window. "
        "Configured EP priority is not proof of actual node placement. `1st infer` is "
        "the very first call after load — a lazy TensorRT build, a cuDNN autotune or a "
        "CUDA context lands there. `drift` compares the last "
        "quarter of the run against the first; read it next to `tj max °C` in the "
        "footprint table. It is only meaningful on a run long enough to heat the "
        "board — see `--duration-s`.",
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
        "## Control-rate fit",
        "",
        control_summary(runs),
        "",
    ]
    failed = [r for r in runs if r.get("status") != "ok"]
    if failed:
        parts += ["## Failures", ""]
        for r in failed:
            parts += [f"### {r.get('label')}", "", "```",
                      str(r.get("error")), "```", ""]
    return "\n".join(parts)
