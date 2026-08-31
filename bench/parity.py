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
Cosine hides a scale error, so an absolute difference is reported too — and an absolute
difference means nothing without knowing the action range. Different policies use
different action spaces (joystick rates in [-1, 1], normalized joint targets, a 20-dim
ee6d pose), so the difference is normalized against the **reference run's own observed
action range** rather than an assumed [-1, 1]. `max_abs_diff_pct_of_range` is therefore
comparable across models: 1% means one percent of the span the reference policy
actually commands.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class ResultLoadError(ValueError):
    pass


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def attach_comparison_signature(result: dict) -> dict:
    """Persist the exact semantic/input identity a parity verdict requires."""
    saved = result.get("saved_chunks") or {}
    model = result.get("model") or {}
    meta = result.get("meta") or {}
    obs = result.get("obs") or {}
    export = meta.get("export_info") or {}
    indices = saved.get("obs_indices") or []
    inputs = saved.get("input_sha256") or []
    noise = saved.get("noise_sha256") or []
    records = sorted(zip(indices, inputs, noise), key=lambda row: row[0])

    fields = {
        "schema": 1,
        "policy": {
            "model_key": model.get("key"),
            "family": model.get("family"),
            "task": model.get("task"),
            "real_views": model.get("views"),
            "camera_slots": (model.get("cam_slots") or meta.get("n_cam_slots")
                             or meta.get("num_views")),
            "state_dim": model.get("state_dim"),
            "action_dim": (result.get("chunk_shape") or [None, None])[-1],
            "chunk_size": (result.get("chunk_shape") or [None])[0],
            "num_steps": meta.get("num_steps"),
        },
        "observation": {
            "kind": obs.get("kind"),
            "shape_hw": obs.get("hw"),
            "views": obs.get("views"),
            "seed": obs.get("seed"),
            "indices": [row[0] for row in records],
            "input_sha256": [row[1] for row in records],
            "noise_sha256": [row[2] for row in records],
        },
        "preprocessing": {
            "resize": meta.get("resize"),
            "tokenizer": (meta.get("tokenizer_sha256") or meta.get("tokenizer")
                          or export.get("tokenizer_revision")),
            "stats": meta.get("stats_sha256") or export.get("stats_sha256"),
        },
    }
    result["comparison"] = {"fields": fields, "sha256": _canonical_sha256(fields)}
    return result


def _saved_by_index(result: dict) -> dict[int, np.ndarray]:
    saved = result.get("saved_chunks")
    if not isinstance(saved, dict):
        raise ValueError("missing saved_chunks object")
    chunks = saved.get("chunks")
    indices = saved.get("obs_indices")
    inputs = saved.get("input_sha256")
    noise = saved.get("noise_sha256")
    if not all(isinstance(x, list) for x in (chunks, indices, inputs, noise)):
        raise ValueError("saved chunks require lists of chunks, indices, input and noise hashes")
    if not chunks:
        raise ValueError("run saved no action chunks")
    if not (len(chunks) == len(indices) == len(inputs) == len(noise)):
        raise ValueError("saved chunk/index/input/noise lengths differ")
    if len(set(indices)) != len(indices):
        raise ValueError("saved observation indices are not unique")
    return {int(index): np.asarray(chunk, dtype=np.float64)
            for index, chunk in zip(indices, chunks)}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _identity(result: dict) -> tuple[str | None, str | None]:
    comparison = result.get("comparison") or {}
    fields = comparison.get("fields")
    digest = comparison.get("sha256")
    if not isinstance(fields, dict) or not isinstance(digest, str):
        return None, "missing comparison signature"
    try:
        actual = _canonical_sha256(fields)
    except (TypeError, ValueError) as exc:
        return None, f"malformed comparison signature: {exc}"
    if actual != digest:
        return None, "comparison signature hash does not match its fields"
    return digest, None


def compare(ref: dict, cand: dict) -> dict:
    """One explicitly selected reference run against one candidate run."""
    out = {"reference": ref.get("label"), "candidate": cand.get("label")}

    ref_family = (ref.get("model") or {}).get("family")
    cand_family = (cand.get("model") or {}).get("family")
    if ref_family and cand_family and ref_family != cand_family:
        return {
            **out,
            "verdict": "NOT COMPARABLE",
            "reason": f"model family {ref_family} vs {cand_family}",
        }

    ref_sig, ref_error = _identity(ref)
    cand_sig, cand_error = _identity(cand)
    if ref_error or cand_error:
        why = "; ".join(x for x in (
            f"reference: {ref_error}" if ref_error else None,
            f"candidate: {cand_error}" if cand_error else None,
        ) if x)
        return {**out, "verdict": "IDENTITY MISMATCH", "reason": why}
    if ref_sig != cand_sig:
        return {
            **out,
            "verdict": "IDENTITY MISMATCH",
            "reason": "comparison signatures differ",
            "reference_signature": ref_sig,
            "candidate_signature": cand_sig,
        }

    try:
        a_by_index = _saved_by_index(ref)
        b_by_index = _saved_by_index(cand)
    except ValueError as exc:
        return {**out, "verdict": "NO DATA", "reason": str(exc)}
    if set(a_by_index) != set(b_by_index):
        return {
            **out,
            "verdict": "IDENTITY MISMATCH",
            "reason": "saved observation index sets differ",
            "reference_indices": sorted(a_by_index),
            "candidate_indices": sorted(b_by_index),
        }

    indices = sorted(a_by_index)
    a = np.stack([a_by_index[index] for index in indices])
    b = np.stack([b_by_index[index] for index in indices])
    if a.shape != b.shape:
        return {**out, "verdict": "SHAPE MISMATCH",
                "reference_shape": list(a.shape), "candidate_shape": list(b.shape)}

    finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
    both_seeded = bool((ref.get("saved_chunks") or {}).get("noise_injected")
                       and (cand.get("saved_chunks") or {}).get("noise_injected"))
    out["noise_injected_both"] = both_seeded
    out["n_observations"] = len(indices)
    out["obs_indices"] = indices
    out["finite"] = finite

    if not finite:
        return {**out, "verdict": "FAIL", "reason": "action chunks contain NaN or infinity"}

    if both_seeded:
        cos = [cosine(a[i], b[i]) for i in range(len(indices))]
        diff = np.abs(a - b)
        ref_range = float(np.ptp(a)) or 1.0
        out.update({
            "mode": "elementwise (exact signed inputs and seeded noise)",
            "cosine_min": round(float(np.min(cos)), 7),
            "cosine_mean": round(float(np.mean(cos)), 7),
            "max_abs_diff": float(f"{diff.max():.3e}"),
            "mean_abs_diff": float(f"{diff.mean():.3e}"),
            "reference_action_range": round(ref_range, 4),
            "max_abs_diff_pct_of_range": round(float(diff.max()) / ref_range * 100, 3),
            "first_action_max_abs_diff": float(
                f"{np.abs(a[:, 0] - b[:, 0]).max():.3e}"),
        })
        ok = (out["cosine_min"] >= 0.999
              and out["max_abs_diff_pct_of_range"] <= 1.0)
        out["verdict"] = "PASS" if ok else "FAIL"
    else:
        out["mode"] = "distribution only (noise not injectable in one backend)"
        out["caveat"] = "distribution checks cannot certify numerical parity"
        am, bm = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        out["per_dim_mean_ref"] = [round(float(x), 4) for x in am.mean(0)]
        out["per_dim_mean_cand"] = [round(float(x), 4) for x in bm.mean(0)]
        out["per_dim_std_ref"] = [round(float(x), 4) for x in am.std(0)]
        out["per_dim_std_cand"] = [round(float(x), 4) for x in bm.std(0)]
        dmean = np.abs(am.mean(0) - bm.mean(0)).max()
        dstd = np.abs(am.std(0) - bm.std(0)).max()
        out["max_dim_mean_shift"] = round(float(dmean), 4)
        out["max_dim_std_shift"] = round(float(dstd), 4)
        out["verdict"] = ("PLAUSIBLE" if dmean < 0.15 and dstd < 0.15 else "SUSPECT")
    return out


def load_results(paths: list[Path]) -> list[dict]:
    runs = []
    for path in paths:
        if not path.exists():
            raise ResultLoadError(f"result path does not exist: {path}")
        files = [path] if path.is_file() else sorted(path.glob("*.json"))
        if not files:
            raise ResultLoadError(f"no result JSON files found in: {path}")
        for file in files:
            try:
                result = json.loads(file.read_text())
            except Exception as exc:
                raise ResultLoadError(f"malformed result JSON {file}: {exc}") from exc
            if not isinstance(result, dict):
                raise ResultLoadError(f"result JSON is not an object: {file}")
            result["_file"] = str(file)
            runs.append(result)
    return runs


def pick_reference(runs: list[dict], prefer: str | None = None) -> dict | None:
    if not prefer:
        return None
    matches = [run for run in runs if run.get("label") == prefer]
    if len(matches) != 1:
        return None
    ref = matches[0]
    if ref.get("status") != "ok":
        return None
    try:
        _saved_by_index(ref)
    except ValueError:
        return None
    return ref


def parity_report(runs: list[dict], prefer_ref: str | None = None) -> dict:
    if not prefer_ref:
        return {"error": "an explicit --reference label is required"}
    ref = pick_reference(runs, prefer_ref)
    if ref is None:
        return {"error": f"reference {prefer_ref!r} is missing, duplicated, failed, or has no data"}

    comparisons = []
    for run in runs:
        if run is ref:
            continue
        if run.get("status") != "ok":
            comparisons.append({
                "reference": ref.get("label"),
                "candidate": run.get("label"),
                "verdict": "FAILED RUN",
                "reason": run.get("error") or f"status={run.get('status')!r}",
            })
            continue
        comparisons.append(compare(ref, run))

    if not comparisons:
        return {"error": "no candidate run exists for the selected reference"}
    return {
        "reference": ref.get("label"),
        "reference_file": ref.get("_file"),
        "comparisons": comparisons,
    }
