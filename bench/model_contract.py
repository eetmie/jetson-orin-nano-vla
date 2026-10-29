"""Portable model settings resolved on the export PC; no model libraries needed here."""
from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA = "vla-model/1"
FILENAME = "model_spec.json"
PROFILES = {
    ("lerobot", "smolvla"): "smolvla-base",
    ("lerobot", "xvla"): "xvla-base",
    ("lerobot", "evo1"): "evo1-libero",
    ("lerobot", "pi05"): "pi05-base",
    ("openpi", "pi05"): "pi05-libero",
}


def validate(spec: dict) -> dict:
    if spec.get("schema") != SCHEMA:
        raise ValueError(f"unsupported model specification schema: {spec.get('schema')!r}")
    profile = PROFILES.get((spec.get("framework"), spec.get("family")))
    if profile is None or spec.get("profile") != profile:
        raise ValueError("model specification has an unsupported framework/family/profile")
    for key in ("views", "camera_slots", "state_dim", "max_state_dim", "action_dim",
                "max_action_dim", "chunk_size", "num_steps", "fps"):
        value = spec.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"model specification {key} must be positive")
        if key != "fps" and (not isinstance(value, int)):
            raise ValueError(f"model specification {key} must be an integer")
    for real, padded in (("state_dim", "max_state_dim"), ("action_dim", "max_action_dim"),
                         ("views", "camera_slots")):
        if spec[real] > spec[padded]:
            raise ValueError(f"model specification {real} exceeds {padded}")
    keys = spec.get("image_keys")
    if (not isinstance(keys, list) or len(keys) != spec["views"] or
            any(not isinstance(k, str) or not k for k in keys) or len(set(keys)) != len(keys)):
        raise ValueError("model specification must name each camera once, in input order")
    if spec.get("noise_distribution") not in ("normal", "uniform"):
        raise ValueError("model specification has an unsupported noise distribution")
    if not isinstance(spec.get("name"), str) or not spec["name"] or not isinstance(spec.get("source"), dict):
        raise ValueError("model specification must identify the source checkpoint")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or not tasks or any(not isinstance(t, str) or not t.strip() for t in tasks):
        raise ValueError("model specification must contain at least one task")
    return spec


def load(bundle: Path) -> dict | None:
    path = Path(bundle) / FILENAME
    return validate(json.loads(path.read_text())) if path.is_file() else None


def export_info(bundle: Path) -> dict:
    for name in ("export_info.json", "bundle.json"):
        path = Path(bundle) / name
        if path.is_file():
            return json.loads(path.read_text())
    return {}


# The shapes bench/aot/groot/plugin.py still fixes; anything else fails inside the AOT build.
# Camera count, prompt length and the K|V precision follow the bundle's graphs.
GROOT_AOT = {
    "groot-n1.6-split/": {"action_horizon": 50, "image_hw": [252, 252]},
    "groot-n1.7-split/": {"action_horizon": 40, "image_hw": [256, 256]},
}


def _groot_aot_unavailable(info: dict) -> str | None:
    schema = info.get("schema", "")
    want = next((v for k, v in GROOT_AOT.items() if schema.startswith(k)), None)
    if want is None:
        return None
    for key in want:
        if info.get(key) != want[key]:
            return f"GR00T AOT kernels are built for {key} {want[key]}, this bundle has {info.get(key)}"
    return None


def display_name(model: str, info: dict, spec: dict | None = None) -> str:
    """The model's name in labels and results: a custom GR00T checkpoint is named after its
    directory, not after the base profile its runtime shares."""
    if spec is not None:
        return spec["name"]
    checkpoint, base = info.get("checkpoint"), info.get("model")
    if model.startswith("groot-") and checkpoint and base:
        name = Path(checkpoint).name
        if not name.startswith(base.replace("/", "--") + "--"):    # not the pinned Hub download
            return f"{model.removesuffix('-base')}:{name}"
    return model


def aot_unavailable(spec: dict | None, info: dict) -> str | None:
    """Use only recipes whose fixed shapes the PC checked, before building engines."""
    if spec is None:
        return _groot_aot_unavailable(info)  # Other legacy bundles keep their existing checks.
    aot = spec.get("aot", {})
    if not aot.get("supported", False):
        return aot.get("reason") or "no AOT recipe for this model's exported shapes"
    if spec["family"] == "smolvla" and (
            spec["views"] != 2 or spec["chunk_size"] != 50 or
            info.get("n_cam_slots") != 2 or (info.get("kv") or {}).get("cross") != "expert-projected"):
        return "SmolVLA AOT needs two cameras, a 50-action chunk and hoisted cross K/V"
    return None
