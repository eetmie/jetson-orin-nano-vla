"""Cross-backend parity: do these runtimes produce the same actions?

Speed is only interesting if the actions survive. On this board that is a live
question rather than a formality: the Orin Nano is compute 8.7, FP16 is the only
fast reduced precision it has, and blanket FP16 on SmolVLA is exactly the
configuration that collapsed the SigLIP vision tower to cosine 0.805 on Blackwell —
730 constants in the vision attention overflow FP16's exponent range. The TensorRT
path avoids it by keeping layer norms in FP32 and letting rejected ops fall back;
a naive `.half()` does not. So parity is a first-class metric here, not a footnote.

How the comparison is possible at all
-------------------------------------
SmolVLA's action head integrates a flow-matching ODE from a random starting point.
Two runs on the identical image disagree simply because they drew different noise.
`bench/obs.py` therefore hands every backend the same seeded noise for observation
`i`, so chunks line up element by element and cosine means what it looks like it
means. Backends that cannot accept injected noise — an HTTP server that samples its
own — are reported separately under a distribution comparison, which can catch a
gross failure (wrong scale, saturated output, dead dimension) but cannot certify
numerical parity. That limitation is stated in the output rather than papered over.

Scale
-----
Excavator actions are joystick rate commands in [-1, 1], so `max_abs_diff` is read
directly as a fraction of full stick deflection: 0.02 is 1% of full travel, 0.2 is a
different move. Cosine alone hides that, which is why both are reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FULL_SCALE = 2.0        # actions live in [-1, 1]


def _chunks(result: dict) -> np.ndarray | None:
    saved = result.get("saved_chunks") or {}
    ch = saved.get("chunks")
    return np.asarray(ch, dtype=np.float64) if ch else None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else float("nan")


def compare(ref: dict, cand: dict) -> dict:
    """One reference run against one candidate run."""
    a, b = _chunks(ref), _chunks(cand)
    out = {"reference": ref.get("label"), "candidate": cand.get("label")}
    if a is None or b is None:
        return {**out, "verdict": "NO DATA", "reason": "a run saved no action chunks"}

    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.shape != b.shape:
        return {**out, "verdict": "SHAPE MISMATCH",
                "reference_shape": list(a.shape), "candidate_shape": list(b.shape)}

    both_seeded = (ref.get("saved_chunks", {}).get("noise_injected")
                   and cand.get("saved_chunks", {}).get("noise_injected"))
    out["noise_injected_both"] = bool(both_seeded)
    out["n_observations"] = n
    out["finite"] = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))

    if both_seeded:
        cos = [cosine(a[i], b[i]) for i in range(n)]
        diff = np.abs(a - b)
        out.update({
            "mode": "elementwise (identical seeded noise)",
            "cosine_min": round(float(np.min(cos)), 7),
            "cosine_mean": round(float(np.mean(cos)), 7),
            "max_abs_diff": float(f"{diff.max():.3e}"),
            "mean_abs_diff": float(f"{diff.mean():.3e}"),
            "max_abs_diff_pct_full_scale": round(float(diff.max()) / FULL_SCALE * 100, 3),
            "first_action_max_abs_diff": float(f"{np.abs(a[:, 0] - b[:, 0]).max():.3e}"),
        })
        # 0.999 is the threshold the on-device guard has used since the Spark sweep;
        # 1% of full stick is the "would you feel it on the machine" line.
        ok = (out["cosine_min"] >= 0.999
              and out["max_abs_diff_pct_full_scale"] <= 1.0
              and out["finite"])
        out["verdict"] = "PASS" if ok else "FAIL"
    else:
        # No shared noise: only distribution-level checks are honest.
        out["mode"] = "distribution only (noise not injectable in one backend)"
        out["caveat"] = ("cannot certify numerical parity — the two runs integrated "
                         "different noise draws. A PASS here means 'not obviously "
                         "broken', not 'matches'.")
        am, bm = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        out["per_dim_mean_ref"] = [round(float(x), 4) for x in am.mean(0)]
        out["per_dim_mean_cand"] = [round(float(x), 4) for x in bm.mean(0)]
        out["per_dim_std_ref"] = [round(float(x), 4) for x in am.std(0)]
        out["per_dim_std_cand"] = [round(float(x), 4) for x in bm.std(0)]
        dmean = np.abs(am.mean(0) - bm.mean(0)).max()
        dstd = np.abs(am.std(0) - bm.std(0)).max()
        out["max_dim_mean_shift"] = round(float(dmean), 4)
        out["max_dim_std_shift"] = round(float(dstd), 4)
        out["verdict"] = ("PLAUSIBLE" if out["finite"] and dmean < 0.15 and dstd < 0.15
                          else "SUSPECT")
    return out


def load_results(paths: list[Path]) -> list[dict]:
    runs = []
    for p in paths:
        for f in ([p] if p.is_file() else sorted(p.glob("*.json"))):
            try:
                r = json.loads(f.read_text())
            except Exception:
                continue
            r["_file"] = str(f)
            runs.append(r)
    return runs


def pick_reference(runs: list[dict], prefer: str | None = None) -> dict | None:
    ok = [r for r in runs if r.get("status") == "ok" and _chunks(r) is not None]
    if not ok:
        return None
    if prefer:
        for r in ok:
            if r.get("label") == prefer:
                return r
    # Default gold: full-precision PyTorch — the dtype the model was trained in is
    # closest to, and the only run with no export step between it and the weights.
    for r in ok:
        m = r.get("meta", {})
        if (r.get("backend") == "torch" and m.get("weights_dtype") == "float32"
                and m.get("autocast", "off") == "off"):
            return r
    for r in ok:
        if r.get("backend") == "torch":
            return r
    return ok[0]


def parity_report(runs: list[dict], prefer_ref: str | None = None) -> dict:
    ref = pick_reference(runs, prefer_ref)
    if ref is None:
        return {"error": "no successful run with saved chunks"}
    return {
        "reference": ref.get("label"),
        "reference_file": ref.get("_file"),
        "comparisons": [compare(ref, r) for r in runs
                        if r is not ref and r.get("status") == "ok"],
    }
