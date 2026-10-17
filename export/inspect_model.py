#!/usr/bin/env python3
"""Inspect a checkpoint on the host PC without loading its weights or importing Torch.

    python3 export/inspect_model.py SOURCE [--revision REV] [--openpi-config NAME]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.model_contract import PROFILES, SCHEMA, validate

DEFAULT_TASK = "pick up the object and place it in the box"
FAMILIES = {"smolvla": "num_steps", "xvla": "num_denoising_steps",
            "evo1": "num_inference_timesteps", "pi05": "num_inference_steps"}


@dataclass(frozen=True)
class Source:
    kind: str
    location: str
    revision: str | None = None
    subfolder: str = ""


def subfolder_path(value: str) -> str:
    value = unquote(value).strip("/")
    if any(p in (".", "..") for p in value.split("/")) or "\\" in value:
        raise ValueError("checkpoint subfolder must stay inside the model repository")
    return value


def parse_source(value: str, revision: str | None = None, subfolder: str = "") -> Source:
    path = Path(value).expanduser()
    if path.exists():
        if revision or subfolder:
            raise ValueError("--revision and --subfolder apply to Hub sources only")
        if path.is_file():
            if path.name not in ("config.json", "model.safetensors", "export_config.json"):
                raise ValueError("pass a complete checkpoint directory, not a standalone weights file")
            path = path.parent
        if not (path / "config.json").is_file() and (path / "pretrained_model/config.json").is_file():
            path = path / "pretrained_model"
        return Source("local", str(path.resolve()))
    if value.startswith(("/", "./", "../", "~")):
        raise ValueError(f"local checkpoint does not exist: {value}")
    url = urlparse(value)
    if url.scheme:
        if url.scheme != "https" or url.hostname not in ("huggingface.co", "hf.co") or url.username or url.password:
            raise ValueError("expected a local checkpoint, Hugging Face model id, or https://huggingface.co/... URL")
        parts = url.path.strip("/").split("/")
        if len(parts) < 2 or parts[0] in ("datasets", "spaces"):
            raise ValueError("expected a Hugging Face model repository URL")
        value = "/".join(parts[:2])
        if len(parts) > 2:
            if parts[2] not in ("tree", "blob", "resolve") or len(parts) < 4:
                raise ValueError("use a model repository or tree URL")
            url_revision = unquote(parts[3])
            if revision and revision != url_revision:
                raise ValueError("URL revision conflicts with --revision")
            revision = url_revision
            url_folder = "/".join(parts[4:] if parts[2] == "tree" else parts[4:-1])
            if subfolder and subfolder_path(subfolder) != subfolder_path(url_folder):
                raise ValueError("URL subfolder conflicts with --subfolder")
            subfolder = url_folder
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value) or ".." in value:
        raise ValueError(f"invalid model source {value!r}; use OWNER/MODEL or an existing directory")
    return Source("hub", value, revision, subfolder_path(subfolder))


def json_file(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def resolve_metadata(source: Source) -> tuple[Path, Source]:
    if source.kind == "local":
        return Path(source.location), source
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ValueError("Hub inspection needs huggingface_hub; use ./export.sh SOURCE --inspect (it prepares the small Hub environment)") from exc
    try:
        info = HfApi().model_info(source.location, revision=source.revision or "main")
    except Exception as exc:
        raise ValueError(f"cannot inspect Hub model {source.location!r}: {exc}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError("Hub did not return an immutable model revision")
    resolved = Source("hub", source.location, info.sha, source.subfolder)
    names = {item.rfilename for item in info.siblings}
    prefix = source.subfolder + "/" if source.subfolder else ""
    if prefix + "config.json" not in names and prefix + "pretrained_model/config.json" in names:
        prefix += "pretrained_model/"
        resolved = Source("hub", source.location, info.sha, prefix.rstrip("/"))
    wanted = [name for name in names if name.startswith(prefix) and (
        name[len(prefix):] in ("config.json", "export_config.json", "train_config.json",
                              "policy_preprocessor.json", "policy_postprocessor.json", "adapter_config.json")
        or (name[len(prefix):].startswith("assets/") and name.endswith("/norm_stats.json")))]
    if not wanted and not any(name.startswith(prefix + "params/") for name in names):
        raise ValueError("repository does not contain checkpoint configuration; select its checkpoint with --subfolder")
    # Only small JSON metadata is fetched here. Full weights follow after inspection.
    root = None
    for name in sorted(wanted):
        try:
            file = Path(hf_hub_download(source.location, name, revision=info.sha))
        except Exception as exc:
            raise ValueError(f"cannot read checkpoint metadata {name!r}: {exc}") from exc
        root = file.parents[len(PurePosixPath(name).parts) - 1]
    if root is None:
        raise ValueError("OpenPI checkpoint has no export_config.json; provide the training/export configuration")
    return root / resolved.subfolder, resolved


def dimension(features: dict, key: str) -> int:
    shape = (features.get(key) or {}).get("shape")
    if not isinstance(shape, list) or len(shape) != 1 or not isinstance(shape[0], int) or shape[0] <= 0:
        raise ValueError(f"checkpoint must declare a one-dimensional {key} feature")
    return shape[0]


def checkpoint_header(checkpoint: Path) -> dict:
    """Read only safetensors headers, including shards. Never load tensor bytes."""
    index = json_file(checkpoint / "model.safetensors.index.json", {})
    files = sorted(set(index.get("weight_map", {}).values())) or ["model.safetensors"]
    header = {}
    for name in files:
        path = checkpoint / name
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
            raise ValueError("weights index points outside the checkpoint")
        if not path.is_file():
            if index:
                raise ValueError(f"checkpoint is missing weights shard {name}")
            continue
        with path.open("rb") as stream:
            length = stream.read(8)
            if len(length) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            size = struct.unpack("<Q", length)[0]
            if size > 16 * 1024**2 or size > path.stat().st_size - 8:
                raise ValueError(f"invalid safetensors header size: {path}")
            try:
                tensors = json.loads(stream.read(size))
            except ValueError as exc:
                raise ValueError(f"invalid safetensors header: {path}") from exc
            header.update({k: v for k, v in tensors.items() if k != "__metadata__"})
    return header


def inspect_checkpoint(checkpoint: Path, *, source: Source | None = None, views: int | None = None,
                       tasks: list[str] | None = None, fps: float | None = None,
                       openpi_config: str | None = None, config_file: Path | None = None) -> dict:
    checkpoint = Path(checkpoint)
    config = json_file(checkpoint / "config.json", {})
    recipe = json_file(config_file or checkpoint / "export_config.json", {})
    if (checkpoint / "adapter_config.json").exists() and not (checkpoint / "model.safetensors").exists():
        raise ValueError("adapter-only checkpoint: merge the adapter into full model weights before exporting")
    family = config.get("type")
    framework = recipe.get("framework") or ("lerobot" if family else "openpi")
    if framework == "lerobot":
        if family not in FAMILIES:
            raise ValueError(f"unsupported LeRobot policy {family!r}; supported: {', '.join(FAMILIES)}")
        inputs = config.get("input_features") or {}
        cameras = {k: v for k, v in inputs.items() if v.get("type") == "VISUAL" or k.startswith("observation.images.")}
        state_dim = dimension(inputs, "observation.state")
        action_dim = dimension(config.get("output_features") or {}, "action")
        fields = {"state_dim": state_dim, "action_dim": action_dim,
                  "max_state_dim": config.get("max_state_dim", state_dim),
                  "max_action_dim": config.get("max_action_dim", action_dim),
                  "chunk_size": config.get("chunk_size"), "num_steps": config.get(FAMILIES[family])}
        # X-VLA's model width is selected by its action mode, not output_features.
        if family == "xvla":
            modes = {"ee6d": 20, "joint": 14, "agibot_ee6d": 20, "franka_joint7": 20,
                     "so101_bimanual": 20, "auto": config.get("max_action_dim", 20)}
            action_mode = str(config.get("action_mode", "")).lower()
            if action_mode not in modes:
                raise ValueError(f"unsupported X-VLA action_mode {action_mode!r}")
            fields["max_action_dim"] = modes[action_mode]
        image_keys = list(cameras)
        shapes = {k: v.get("shape") for k, v in cameras.items()}
    elif framework == "openpi":
        family = recipe.get("family", "pi05")
        if family != "pi05":
            raise ValueError("the OpenPI exporter currently supports pi05 checkpoints")
        name = openpi_config or recipe.get("openpi_config")
        if not name:
            raise ValueError("OpenPI needs --openpi-config NAME or export_config.json with openpi_config and observation settings")
        # LIBERO is a named, fixed upstream recipe; custom recipes declare their observation contract.
        defaults = ({"state_dim": 8, "action_dim": 7, "max_state_dim": 32, "max_action_dim": 32,
                     "chunk_size": 10, "num_steps": 10,
                     "image_keys": ["base_0_rgb", "left_wrist_0_rgb"],
                     "raw_image_keys": ["observation/image", "observation/wrist_image"],
                     "state_key": "observation/state", "discrete_state_input": False}
                    if name == "pi05_libero" else {})
        settings = {**defaults, **recipe}
        asset_id = settings.get("asset_id")
        assets = ([checkpoint / "assets" / asset_id / "norm_stats.json"] if asset_id else
                  list((checkpoint / "assets").rglob("norm_stats.json")))
        if len(assets) == 1:
            stats = json_file(assets[0], {}).get("norm_stats", {})
            for field, key in (("state_dim", "state"), ("action_dim", "actions")):
                entry = stats.get(key, {})
                vector = entry.get("mean", entry.get("q01"))
                if field not in settings and isinstance(vector, list) and vector:
                    settings[field] = len(vector)
        if "max_state_dim" not in settings and config.get("action_dim"):
            settings["max_state_dim"] = config["action_dim"]
        fields = {k: settings.get(k, config.get("action_horizon" if k == "chunk_size" else
                                             "action_dim" if k == "max_action_dim" else k))
                  for k in ("state_dim", "action_dim", "max_state_dim", "max_action_dim", "chunk_size", "num_steps")}
        image_keys = settings.get("image_keys", [])
        shapes = {k: [3, 224, 224] for k in image_keys}
        recipe = {**settings, "openpi_config": name}
        if not image_keys or any(v is None for v in fields.values()):
            raise ValueError("custom OpenPI checkpoints need export_config.json declaring image_keys, state/action dimensions, chunk_size and num_steps")
        raw_keys = recipe.get("raw_image_keys")
        if not isinstance(raw_keys, list) or len(raw_keys) != len(image_keys) or not recipe.get("state_key"):
            raise ValueError("OpenPI export_config.json must declare raw_image_keys and state_key for stock-policy fixtures")
    else:
        raise ValueError(f"unsupported checkpoint framework {framework!r}")
    count = len(image_keys) if views is None else views
    if not isinstance(count, int) or not 1 <= count <= len(image_keys):
        raise ValueError(f"--views must be between 1 and the checkpoint's {len(image_keys)} declared cameras")
    if framework == "openpi" and count != len(image_keys):
        raise ValueError("OpenPI views are fixed by its input transforms; update export_config.json and training configuration together")
    for key, shape in shapes.items():
        if (not isinstance(shape, list) or len(shape) != 3 or shape[0] != 3 or
                any(not isinstance(d, int) or d <= 0 for d in shape)):
            raise ValueError(f"{key} must declare an RGB [3, height, width] image shape")
    if not tasks:
        tasks = recipe.get("tasks") or [DEFAULT_TASK]
    spec = {"schema": SCHEMA, "framework": framework, "family": family,
            "profile": PROFILES[(framework, family)], "name": checkpoint.name,
            "source": asdict(source or Source("local", str(checkpoint.resolve()))),
            "image_keys": image_keys[:count], "image_shapes": {k: shapes[k] for k in image_keys[:count]},
            "views": count, "camera_slots": count, **fields, "tasks": tasks,
            "fps": fps if fps is not None else recipe.get("fps", 30),
            "noise_distribution": "uniform" if family == "evo1" else "normal",
            "processors": {k: json_file(checkpoint / f"policy_{k}.json", {}) for k in ("preprocessor", "postprocessor")}}
    if framework == "openpi":
        spec["openpi"] = recipe
    else:
        spec["architecture"] = {k: config[k] for k in ("vlm_model_name", "resize_imgs_with_padding",
            "num_vlm_layers", "num_expert_layers",
            "expert_width_multiplier", "hidden_size", "num_heads", "depth", "action_mode") if k in config}
        spec["architecture"]["image_resolution"] = config.get("resize_imgs_with_padding", config.get("image_resolution"))
    validate(spec)
    header = checkpoint_header(checkpoint)
    for key, tensor in header.items():
        expected = None
        if key.endswith("state_proj.weight"):
            expected = spec["max_state_dim"]
        if key.endswith("action_in_proj.weight"):
            expected = spec["max_action_dim"]
        shape = tensor.get("shape")
        if expected is not None and (not isinstance(shape, list) or len(shape) != 2 or shape[-1] != expected):
            raise ValueError(f"weight {key} shape disagrees with checkpoint configuration ({expected})")
    spec["inspection"] = {"weight_tensors_checked": len(header),
                          "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--revision")
    parser.add_argument("--subfolder", default="")
    parser.add_argument("--views", type=int)
    parser.add_argument("--task", action="append")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--openpi-config")
    parser.add_argument("--config-file", type=Path)
    args = parser.parse_args(argv)
    try:
        checkpoint, source = resolve_metadata(parse_source(args.source, args.revision, args.subfolder))
        spec = inspect_checkpoint(checkpoint, source=source, views=args.views, tasks=args.task,
                                  fps=args.fps, openpi_config=args.openpi_config, config_file=args.config_file)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
