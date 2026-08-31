"""Immutable X-VLA bundle identity and physical processor contracts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


SCHEMA_VERSION = 2
_PROCESSOR_JSONS = ("policy_preprocessor.json", "policy_postprocessor.json")
_TOKENIZER_PATTERNS = (
    "tokenizer*",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: str | Path) -> str:
    """Hash relative names, sizes, and bytes so a local tree has one identity."""
    root = Path(path)
    digest = hashlib.sha256()
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = file.relative_to(root).as_posix().encode()
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        digest.update(file.stat().st_size.to_bytes(8, "big"))
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required processor artifact is missing: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid processor artifact {path}: {exc}") from exc


def _step(document: dict, registry_name: str, source: Path) -> dict:
    matches = [
        step for step in document.get("steps", [])
        if step.get("registry_name") == registry_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{source} must contain exactly one {registry_name!r} step, got "
            f"{len(matches)}")
    return matches[0]


def _feature(features: dict, feature_type: str, where: str) -> tuple[str, dict]:
    matches = [
        (name, spec) for name, spec in features.items()
        if str(spec.get("type", "")).upper() == feature_type
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{where} must contain exactly one {feature_type} feature, got "
            f"{[name for name, _ in matches]}")
    return matches[0]


def _normalization(step: dict, feature_key: str, feature_type: str,
                   checkpoint: Path) -> dict:
    config = step.get("config") or {}
    mode = str((config.get("norm_map") or {}).get(feature_type, "IDENTITY")).upper()
    result = {"mode": mode, "declared_mode": mode,
              "eps": float(config.get("eps", 1e-8)), "stats_available": False}
    if mode == "IDENTITY":
        return result
    if mode != "MEAN_STD":
        raise ValueError(
            f"bundle contract currently supports IDENTITY and MEAN_STD, got "
            f"{feature_type}={mode}")

    state_name = step.get("state_file")
    if not state_name:
        # LeRobot treats a declared mode with no loaded stats as identity. Preserve
        # that behavior, but mark the physical boundary incomplete.
        result["mode"] = "IDENTITY"
        return result
    state_path = checkpoint / state_name
    if not state_path.is_file():
        raise ValueError(f"processor state file is missing: {state_path}")

    from safetensors import safe_open

    with safe_open(state_path, framework="np") as state:
        keys = set(state.keys())
        wanted = [f"{feature_key}.mean", f"{feature_key}.std"]
        missing = [key for key in wanted if key not in keys]
        if missing:
            raise ValueError(f"{state_path} is missing required tensors {missing}")
        mean = state.get_tensor(wanted[0])
        std = state.get_tensor(wanted[1])
    if mean.ndim != 1 or std.shape != mean.shape:
        raise ValueError(
            f"{feature_key} mean/std must be matching vectors, got "
            f"{mean.shape} and {std.shape}")
    result.update({
        "stats_available": True,
        "mean": mean.astype("float32").tolist(),
        "std": std.astype("float32").tolist(),
        "state_file": f"processor/{state_path.name}",
        "state_file_sha256": sha256_file(state_path),
    })
    return result


def build_processor_contract(checkpoint: str | Path, config: dict | None = None) -> dict:
    """Extract the real robot boundary from a saved LeRobot policy processor."""
    checkpoint = Path(checkpoint)
    cfg = config or _load_json(checkpoint / "config.json")
    pre_path = checkpoint / "policy_preprocessor.json"
    post_path = checkpoint / "policy_postprocessor.json"
    pre = _load_json(pre_path)
    post = _load_json(post_path)

    input_features = cfg.get("input_features") or {}
    output_features = cfg.get("output_features") or {}
    state_key, state_spec = _feature(input_features, "STATE", "input_features")
    action_key, action_spec = _feature(output_features, "ACTION", "output_features")
    state_dim = int(state_spec["shape"][-1])
    action_dim = int(action_spec["shape"][-1])
    model_state_dim = int(cfg.get("max_state_dim") or state_dim)
    model_action_dim = int(cfg.get("max_action_dim") or action_dim)
    if state_dim > model_state_dim or action_dim > model_action_dim:
        raise ValueError(
            f"real dimensions state/action={state_dim}/{action_dim} exceed model "
            f"dimensions {model_state_dim}/{model_action_dim}")

    pre_step = _step(pre, "normalizer_processor", pre_path)
    post_step = _step(post, "unnormalizer_processor", post_path)
    state_norm = _normalization(pre_step, state_key, "STATE", checkpoint)
    action_norm = _normalization(post_step, action_key, "ACTION", checkpoint)
    if len(state_norm.get("mean", [0] * state_dim)) != state_dim:
        raise ValueError("state normalization statistics do not match the state feature")
    if len(action_norm.get("mean", [0] * action_dim)) != action_dim:
        raise ValueError("action normalization statistics do not match the action feature")

    tokenizer_step = _step(pre, "tokenizer_processor", pre_path)
    tokenizer_cfg = tokenizer_step.get("config") or {}
    tokenizer_len = int(tokenizer_cfg.get("max_length") or 0)
    if tokenizer_len <= 0:
        raise ValueError(f"{pre_path} has an invalid tokenizer max_length")

    artifact_names = set(_PROCESSOR_JSONS)
    for step in (pre_step, post_step):
        if step.get("state_file"):
            artifact_names.add(step["state_file"])
    artifacts = {}
    for name in sorted(artifact_names):
        source = checkpoint / name
        if not source.is_file():
            raise ValueError(f"required processor artifact is missing: {source}")
        artifacts[f"processor/{name}"] = sha256_file(source)

    physical_boundary_complete = all(
        norm["declared_mode"] == "IDENTITY" or norm["stats_available"]
        for norm in (state_norm, action_norm)
    )
    return {
        "version": 1,
        "physical_boundary_complete": physical_boundary_complete,
        "input_features": input_features,
        "output_features": output_features,
        "state": {
            "feature": state_key,
            "dim": state_dim,
            "model_dim": model_state_dim,
            "normalization": state_norm,
        },
        "action": {
            "feature": action_key,
            "dim": action_dim,
            "model_dim": model_action_dim,
            "normalization": action_norm,
        },
        "action_mode": cfg.get("action_mode"),
        "tokenizer": {
            "source": tokenizer_cfg.get("tokenizer_name"),
            "max_length": tokenizer_len,
            "padding": tokenizer_cfg.get("padding"),
            "padding_side": tokenizer_cfg.get("padding_side"),
            "truncation": tokenizer_cfg.get("truncation"),
        },
        "artifacts": artifacts,
    }


def copy_processor_artifacts(checkpoint: str | Path, out: str | Path,
                             contract: dict) -> None:
    checkpoint, out = Path(checkpoint), Path(out)
    target = out / "processor"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for relative in contract["artifacts"]:
        name = Path(relative).name
        shutil.copy2(checkpoint / name, target / name)


def materialize_tokenizer(source: str | Path, out: str | Path) -> dict:
    """Save the exact tokenizer files into the bundle for offline-only loading."""
    source_path = Path(source)
    revision = None
    resolved = source_path
    if not source_path.exists():
        from huggingface_hub import snapshot_download

        resolved = Path(snapshot_download(
            repo_id=str(source),
            allow_patterns=list(_TOKENIZER_PATTERNS),
        ))
        revision = resolved.name

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(resolved), local_files_only=True)
    target = Path(out) / "tokenizer"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    tokenizer.save_pretrained(target)
    return {
        "source": str(source),
        "revision": revision,
        "path": "tokenizer",
        "tree_sha256": tree_sha256(target),
    }


def verify_bundle(split_dir: str | Path, *, verify_manifest: bool = False) -> dict:
    """Fail closed for schema-v2 bundles before any TensorRT engine is constructed."""
    split_dir = Path(split_dir)
    bundle = _load_json(split_dir / "bundle.json")
    if int(bundle.get("schema_version") or 1) < SCHEMA_VERSION:
        return bundle

    checkpoint = bundle.get("checkpoint") or {}
    if not checkpoint.get("tree_sha256"):
        raise ValueError("schema-v2 bundle has no checkpoint tree_sha256")
    if checkpoint.get("random_init"):
        raise ValueError("random-init smoke graphs are not a deployable bundle")
    contract = bundle.get("processor_contract") or {}
    if int(contract.get("version") or 0) != 1:
        raise ValueError("schema-v2 bundle has no supported processor contract")

    for relative, expected in contract.get("artifacts", {}).items():
        path = split_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"processor artifact identity mismatch: {path}")

    tokenizer = bundle.get("tokenizer") or {}
    tokenizer_path = split_dir / str(tokenizer.get("path") or "")
    if not tokenizer_path.is_dir():
        raise ValueError("schema-v2 bundle has no local tokenizer directory")
    if tree_sha256(tokenizer_path) != tokenizer.get("tree_sha256"):
        raise ValueError("local tokenizer tree does not match bundle identity")

    for graph in bundle.get("graphs") or []:
        path = split_dir / str(graph.get("file") or "")
        if not path.is_file():
            raise ValueError(f"bundle graph is missing: {path}")

    if verify_manifest:
        manifest = split_dir / "MANIFEST.sha256"
        if not manifest.is_file():
            raise ValueError("schema-v2 bundle has no MANIFEST.sha256")
        for line in manifest.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            path = split_dir / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"manifest identity mismatch: {path}")
    return bundle


def normalize_vector(value, contract: dict):
    import numpy as np

    array = np.asarray(value, dtype=np.float32)
    norm = contract["normalization"]
    if norm["mode"] == "IDENTITY":
        return array
    if norm["mode"] == "MEAN_STD":
        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        return (array - mean) / (std + float(norm.get("eps", 1e-8)))
    raise ValueError(f"unsupported normalization mode {norm['mode']!r}")


def unnormalize_vector(value, contract: dict):
    import numpy as np

    array = np.asarray(value, dtype=np.float32)
    norm = contract["normalization"]
    if norm["mode"] == "IDENTITY":
        return array
    if norm["mode"] == "MEAN_STD":
        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        return array * std + mean
    raise ValueError(f"unsupported normalization mode {norm['mode']!r}")
