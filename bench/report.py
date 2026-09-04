"""Turn a directory of run JSONs into the markdown tables that go in RESULTS.md."""

from __future__ import annotations

from pathlib import Path

from .parity import compare, load_results


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


def validity_table(runs: list[dict]) -> str:
    rows = []
    for run in runs:
        validity = run.get("validity") or {}
        deployable = _g(run, "model", "deployable", default=True)
        rows.append([
            run.get("label"),
            validity.get("execution"),
            validity.get("instrumentation"),
            validity.get("placement"),
            validity.get("provenance"),
            "yes" if deployable else "**no**",
        ])
    return _md_table(rows, [
        "run", "execution", "instrumentation", "placement", "provenance",
        "deployable",
    ])


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


def _parity_pairs(runs: list[dict]) -> list[dict]:
    """Every run pair `compare` accepts.

    `compare` refuses anything whose comparison signature differs -- different policy,
    different observations, different injected noise -- so a rejected pair is dropped
    rather than reported with a caveat. PyTorch is preferred as the reference where a
    torch run exists: it is the unquantised side, so the difference reads as the cost
    of the conversion.
    """
    def is_torch(r: dict) -> bool:
        return str(r.get("backend") or "").startswith("torch")

    ok = [r for r in runs if r.get("status") == "ok"]
    out, seen = [], set()
    for ref in sorted(ok, key=lambda r: (not is_torch(r), r.get("label") or "")):
        for cand in ok:
            if cand is ref:
                continue
            key = frozenset((ref.get("label"), cand.get("label")))
            if key in seen:
                continue
            result = compare(ref, cand)
            if not str(result.get("mode", "")).startswith("elementwise"):
                continue
            seen.add(key)
            out.append(result)
    return out


def parity_table(runs: list[dict], pairs: list[dict]) -> str:
    """One row per model: does the deployed export still produce the reference actions?

    Deliberately one line each. The per-boundary and per-pair detail is in the run
    JSONs for anyone who needs it; what belongs in a summary is whether the converted
    model can be used. The reported difference is on the FIRST action of the chunk,
    because that is the one a control loop executes before the next inference lands.
    """
    by_label = {r.get("label"): r for r in runs}
    order, models = [], {}
    for r in runs:
        key = _g(r, "model", "key", default=None) or r.get("label")
        if key not in models:
            order.append(key)
            models[key] = []
        models[key].append(r)

    rows = []
    for key in order:
        row = None
        for pair in pairs:
            cand = by_label.get(pair["candidate"], {})
            if _g(cand, "model", "key", default=None) != key:
                continue
            ref = by_label.get(pair["reference"], {})
            dtype = _g(ref, "meta", "weights_dtype", default="") or ""
            span = pair.get("reference_action_range") or 1.0
            first = pair.get("first_action_max_abs_diff")
            row = [key, f"PyTorch {dtype} on this board".rstrip(),
                   pair.get("cosine_min"), first,
                   round(float(first) / float(span) * 100, 2)]
            break
        if row is None:
            # No PyTorch run to compare against, so fall back to the fixture the bundle
            # carries. Only the action boundary is summarised: it is the policy output,
            # and the intermediate boundaries are in the run JSON.
            for r in models[key]:
                action = _g(r, "meta", "fixture_parity", "reports", "action",
                            default=None)
                if action:
                    row = [key, "native fixture inside the bundle",
                           round(action["cosine"], 7),
                           float(f"{action['max_abs']:.3g}"), None]
                    break
        if row is None:
            row = [key, "not measured on this board", None, None, None]
        rows.append(row)
    return _md_table(rows, [
        "model", "checked against", "cosine",
        "max abs diff, executed action", "% of action range"])


def _full_chunk_note(runs: list[dict], pairs: list[dict]) -> str:
    """Say how much worse the full chunk is than its first action, from the data.

    Written as a sentence rather than a column because it is a caveat on the table
    above, and because hardcoding the figure into generated prose is exactly the drift
    this file exists to avoid.
    """
    by_label = {r.get("label"): r for r in runs}
    worst = []
    for pair in pairs:
        whole = pair.get("max_abs_diff_pct_of_range")
        span = pair.get("reference_action_range") or 1.0
        first = float(pair.get("first_action_max_abs_diff") or 0.0) / float(span) * 100
        if whole is None or whole <= first * 1.5:
            continue
        key = _g(by_label.get(pair["candidate"], {}), "model", "key",
                 default=pair["candidate"])
        chunk = _g(by_label.get(pair["candidate"], {}), "meta", "chunk_size",
                   default=None)
        worst.append(f"{key}'s worst step over the full "
                     f"{f'{chunk:g}' if chunk else 'chunk'} is {whole:g}% of range")
    if not worst:
        return ""
    return "Later steps in a long chunk drift further: " + "; ".join(worst) + ". "


def build_report(paths: list[Path]) -> str:
    runs = load_results(paths)
    runs.sort(key=lambda r: (r.get("status") != "ok", r.get("label") or ""))
    if not runs:
        return "_no result JSONs found_"
    pairs = _parity_pairs(runs)

    parts = [
        "# Results",
        "",
        "Generated by `python -m bench report`. Every row is one run JSON in "
        "`results/`; the JSON holds far more than fits here.",
        "",
        "## Validity",
        "",
        validity_table(runs),
        "",
        "Configured EP priority is not measured node placement. A nondeployable "
        "result is infrastructure evidence only and must not control a robot. Parity "
        "is not a column here: see the measured values below.",
        "",
        "## Parity",
        "",
        "Does the converted model still produce the reference actions? Every backend "
        "is handed the same seeded observations and the same injected noise, so the "
        "action chunks line up element by element.",
        "",
        parity_table(runs, pairs),
        "",
        "The difference is on the **first action of the chunk** — the one a control "
        "loop executes before the next inference lands — and is normalised against the "
        "reference's own observed action range, not an assumed [-1, 1], because these "
        "policies do not share an action space. " + _full_chunk_note(runs, pairs)
        + "Thresholds are in the README; `python -m bench parity --reference <label>` "
        "applies them.",
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
