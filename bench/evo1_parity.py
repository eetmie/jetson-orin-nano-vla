"""Native-fixture parity gate shared by the EVO1 backend and check script."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _compare(expected: np.ndarray, actual: np.ndarray) -> dict:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    if expected_array.shape != actual_array.shape:
        raise ValueError(
            f"fixture shape mismatch: {expected_array.shape} != {actual_array.shape}"
        )
    lhs = expected_array.astype(np.float64, copy=False).reshape(-1)
    rhs = actual_array.astype(np.float64, copy=False).reshape(-1)
    denominator = np.linalg.norm(lhs) * np.linalg.norm(rhs)
    cosine = float(np.dot(lhs, rhs) / denominator) if denominator else float("nan")
    return {
        "cosine": cosine,
        "max_abs": float(np.max(np.abs(lhs - rhs))),
        "mean_abs": float(np.mean(np.abs(lhs - rhs))),
    }


def validate_fixture(policy, bundle_dir: str | Path, threshold: float = 0.999) -> dict:
    """Check stored graph boundaries and the raw-observation preprocessing path."""
    from .vendor.evo1_split_ort import preprocess_image

    root = Path(bundle_dir)
    bundle = policy.bundle
    with np.load(root / bundle["fixture"]["file"], allow_pickle=False) as fixture:
        # raw_images is the full view stack; raw_image is only the first view, kept so
        # older one-view fixtures still load. Preferring the stack means a two-view
        # bundle is never silently checked against one camera -- which would pass the
        # preprocessing comparison on view 0 and certify nothing about view 1.
        raw = fixture["raw_images"] if "raw_images" in fixture else fixture["raw_image"]
        output = policy.run_fixture(fixture)
        valid = np.broadcast_to(
            fixture["context_mask"][..., None], fixture["expected_fused"].shape
        )
        reports = {
            "vision": _compare(fixture["expected_vision"], output["vision"]),
            "fused_valid": _compare(
                fixture["expected_fused"][valid], output["fused"][valid]
            ),
            "action": _compare(fixture["expected_action"], output["action"]),
            "image_preprocess": _compare(
                fixture["pixel_values"],
                preprocess_image(raw, int(bundle["image_size"])),
            ),
        }
        from_raw = policy.sample_actions(
            raw,
            bundle["fixture"]["task"],
            fixture["state"][0],
            fixture["initial_noise"],
        )
        reports["action_from_raw_observation"] = _compare(
            fixture["expected_action"][0], from_raw
        )

    gated = ("vision", "fused_valid", "action", "action_from_raw_observation")
    status = (
        "PASS"
        if all(reports[name]["cosine"] >= threshold for name in gated)
        else "FAIL"
    )
    return {"status": status, "threshold": threshold, "reports": reports}
