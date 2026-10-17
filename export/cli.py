"""Host PC entry point: source -> inspected model contract -> portable ONNX bundle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.inspect_model import Source, inspect_checkpoint, parse_source, resolve_metadata
from bench.model_contract import FILENAME, export_info, validate
from export.vla_common.bundle.manifest import write_manifest

HERE = ROOT / "export"
ALIASES = {"smolvla": ("smolvla", "lerobot/smolvla_base"),
           "xvla": ("xvla", "lerobot/xvla-base"),
           "evo1": ("xvla", "zuoxingdong/evo1_libero"),
           "pi05-base": ("xvla", "lerobot/pi05_base"),
           "groot-n16": ("groot", "nvidia/GR00T-N1.6-3B"),
           "groot-n17": ("groot17", "nvidia/GR00T-N1.7-3B"),
           "pi05-libero": ("docker", None)}
ENVIRONMENTS = {"smolvla": "smolvla", "xvla": "xvla", "evo1": "xvla", "pi05": "xvla"}


def run(command, **kwargs):
    subprocess.run([str(c) for c in command], check=True, **kwargs)


def hub_python() -> Path:
    if importlib.util.find_spec("huggingface_hub") is not None:
        return Path(sys.executable)
    candidates = [HERE / ".venv-hub/bin/python", *sorted(HERE.glob(".venv-*/bin/python"))]
    for python in candidates:
        if python.is_file() and subprocess.run([str(python), "-c", "import huggingface_hub"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            return python
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    if not Path(uv).is_file():
        raise ValueError("Hub inspection needs uv; install uv or use a local checkpoint")
    venv = HERE / ".venv-hub"
    run([uv, "venv", "--quiet", "--python", "3.12", venv])
    run([uv, "pip", "install", "--quiet", "--python", venv / "bin/python",
         "-r", HERE / "requirements-hub.txt"])
    return venv / "bin/python"


def materialize(source: Source, metadata: Path, python: Path) -> Path:
    if source.kind == "local":
        return metadata
    # Reuse downloads from the original exporter whenever their identity is verified.
    destination = HERE / ".checkpoints" / (source.location.replace("/", "--") + "--" + source.revision[:16])
    for candidate in (HERE / ".checkpoints").glob(source.location.replace("/", "--") + "*"):
        marker = candidate / ".vla-hub-snapshot.json"
        if marker.is_file():
            identity = json.loads(marker.read_text())
            if (identity.get("repo"), identity.get("revision")) == (source.location, source.revision):
                selected = candidate / source.subfolder
                if any((selected / name).exists() for name in ("model.safetensors", "model.safetensors.index.json", "params")):
                    destination = candidate
                    break
    prefix = source.subfolder + "/" if source.subfolder else ""
    patterns = ["*.json", "model*.safetensors", "policy_*", "stats*", "tokenizer*",
                "*.model", "*.txt", "*.jinja", "params/*", "assets/*"]
    # Metadata JSON patterns deliberately stay in the selected folder, not sibling checkpoints.
    patterns[0] = "config.json"
    patterns += ["export_config.json", "train_config.json", "model.safetensors.index.json",
                 "preprocessor*.json", "postprocessor*.json", "special_tokens_map.json"]
    command = [python, "-m", "vla_common.hub_download", source.location,
               "--revision", source.revision, "--local-dir", destination]
    for pattern in patterns:
        command += ["--include", prefix + pattern]
    run(command, env={**os.environ, "PYTHONPATH": str(HERE)})
    return destination / source.subfolder


def aot_recipe(spec: dict, info: dict) -> dict:
    supported = False
    arch = spec.get("architecture", {})
    if spec["family"] == "smolvla":
        supported = (spec["views"] == 2 and spec["chunk_size"] == 50 and
                     arch.get("vlm_model_name") == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" and
                     arch.get("image_resolution") == [512, 512] and
                     arch.get("expert_width_multiplier") == 0.75 and arch.get("num_vlm_layers") == 16 and
                     arch.get("num_expert_layers") in (0, 16) and
                     (info.get("kv") or {}).get("cross") == "expert-projected")
    elif spec["family"] == "xvla":
        supported = (spec["views"] == 3 and spec["chunk_size"] == 30 and arch.get("hidden_size") == 1024 and
                     arch.get("num_heads") == 16 and arch.get("depth") == 24 and info.get("lang_len") == 50)
    # Other families retain native TensorRT until their full exported shape contract is validated.
    return {"supported": supported, "reason": None if supported else
            "the exported shapes use native TensorRT; no matching validated AOT recipe"}


def finalize_bundle(out: Path, spec: dict):
    """Check exporter shapes and store the PC's decision under the bundle checksum."""
    info = export_info(out)
    if not info:
        raise ValueError("exporter did not produce bundle metadata")
    for field, exported in (("chunk_size", info.get("chunk_size", info.get("action_horizon"))),
                            ("max_action_dim", info.get("max_action_dim", info.get("action_dim"))),
                            ("num_steps", info.get("num_steps")),
                            ("camera_slots", info.get("n_cam_slots", info.get("image_slots")))):
        if exported is not None and exported != spec[field]:
            raise ValueError(f"exported {field}={exported} disagrees with inspected {spec[field]}")
    spec["aot"] = aot_recipe(spec, info)
    if spec["family"] == "smolvla":
        tokenizer = out / "tokenizer/tokenizer_config.json"
        tokenizer_config = json.loads(tokenizer.read_text()) if tokenizer.is_file() else {}
        spec["runtime"] = {"lean_tokenizer": tokenizer_config.get("tokenizer_class") == "GPT2Tokenizer"}
    if spec["framework"] == "openpi":
        lock = json.loads((HERE / "pi05/source-lock.json").read_text())
        spec["export_environment"] = {k: lock[k] for k in ("openpi_commit", "base_image")}
        image_id = out.with_name(out.name + ".work") / "image-id"
        if image_id.is_file():
            spec["export_environment"]["image_id"] = image_id.read_text().strip()
    validate(spec)
    (out / FILENAME).write_text(json.dumps(spec, indent=2) + "\n")
    write_manifest(out)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="export.sh", description=__doc__)
    parser.add_argument("source", help="checkpoint directory, Hugging Face URL/id, or a built-in alias")
    parser.add_argument("checkpoint", nargs="?", help="legacy alias checkpoint, or output for source-first usage")
    parser.add_argument("out_dir", nargs="?", type=Path, help="legacy alias output directory")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inspect", action="store_true", help="print resolved model settings; fetch metadata only")
    parser.add_argument("--revision", help="Hub commit, tag or branch, resolved once to a full SHA")
    parser.add_argument("--subfolder", default="")
    parser.add_argument("--views", type=int)
    parser.add_argument("--task", action="append")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--openpi-config")
    parser.add_argument("--export-config", type=Path, help="local JSON observation/export recipe")
    parser.add_argument("--openpi-config-file", type=Path, help="explicit local Python module defining register()")
    parser.add_argument("--embodiment")
    parser.add_argument("--vlm-files")
    parser.add_argument("--hoist-cross-kv", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    alias = args.source if args.source in ALIASES else None
    if alias:
        environment, default = ALIASES[alias]
        value = args.checkpoint or default
        out = args.out or args.out_dir or ROOT / "bundles" / alias
    else:
        if args.out_dir or (args.out and args.checkpoint):
            parser.error("use SOURCE [OUT] or SOURCE --out OUT")
        value, out = args.source, args.out or (Path(args.checkpoint) if args.checkpoint else None)
    try:
        if alias in ("groot-n16", "groot-n17") or (alias == "pi05-libero" and not args.checkpoint and not args.inspect):
            if args.inspect:
                raise ValueError("--inspect currently supports LeRobot and OpenPI checkpoints")
            if alias == "pi05-libero":
                if any((args.views, args.task, args.revision, args.openpi_config, args.export_config, args.openpi_config_file)):
                    raise ValueError("pi05-libero exports its locked public recipe; pass a checkpoint source for a custom OpenPI export")
                run([HERE / "pi05/export.sh", out])
                return 0
            if not (HERE / f".venv-{environment}/bin/python").is_file():
                run([HERE / "setup.sh", environment])
            flags = []
            for name in ("revision", "views", "embodiment", "vlm_files"):
                if getattr(args, name) is not None:
                    flags += ["--" + name.replace("_", "-"), str(getattr(args, name))]
            for task in args.task or []:
                flags += ["--task", task]
            run([HERE / "export.sh", value, out, *flags])
            return 0
        if alias == "pi05-libero":
            args.openpi_config = args.openpi_config or "pi05_libero"
            if value is None:  # Metadata-only inspection of the pinned public recipe.
                value = str(HERE / "pi05")
        source = parse_source(value, args.revision, args.subfolder)
        if source.kind == "hub":
            python = hub_python()
            if Path(sys.executable).absolute() != python.absolute():
                os.execv(str(python), [str(python), str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])])
            if alias and not source.revision:
                pins = json.loads((HERE / "hub-lock.json").read_text())
                source = Source("hub", source.location, pins.get(source.location), source.subfolder)
        metadata, source = resolve_metadata(source)
        views = args.views if args.views is not None else (2 if alias == "smolvla" and not args.checkpoint else None)
        spec = inspect_checkpoint(metadata, source=source, views=views, tasks=args.task, fps=args.fps,
                                  openpi_config=args.openpi_config, config_file=args.export_config)
        if spec["framework"] != "openpi" and any((args.openpi_config, args.openpi_config_file, args.export_config)):
            raise ValueError("OpenPI configuration options require an OpenPI checkpoint")
        if args.openpi_config_file and not args.openpi_config_file.is_file():
            raise ValueError(f"OpenPI configuration module does not exist: {args.openpi_config_file}")
        spec["name"] = alias if alias and not args.checkpoint else (Path(source.subfolder).name or source.location.split("/")[-1])
        if args.inspect:
            print(json.dumps(spec, indent=2))
            return 0
        identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
        slug = re.sub(r"[^\w.-]+", "-", spec["name"])
        out = Path(out or ROOT / "bundles" / f"{slug}-{identity}").expanduser().resolve()
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise ValueError(f"output must be new or empty: {out}")
        checkpoint = materialize(source, metadata, hub_python() if source.kind == "hub" else Path(sys.executable))
        # Repeat the header check on the materialized weights, not just remote metadata.
        full = inspect_checkpoint(checkpoint, source=source, views=views, tasks=spec["tasks"], fps=spec["fps"],
                                  openpi_config=args.openpi_config, config_file=args.export_config)
        spec["inspection"] = full["inspection"]
        if not (checkpoint / "model.safetensors").is_file() and not (checkpoint / "model.safetensors.index.json").is_file() and not (checkpoint / "params").is_dir():
            raise ValueError("checkpoint lacks full model weights (model.safetensors, shards, or OpenPI params/)")
        print(f"{spec['framework']}/{spec['family']}: {spec['views']} cameras, state {spec['state_dim']}, "
              f"action {spec['action_dim']} (padded {spec['max_action_dim']}), chunk {spec['chunk_size']}", flush=True)
        if spec["framework"] == "openpi":
            from export.openpi_export import export_openpi
            export_openpi(checkpoint, out, spec, args.openpi_config_file)
        else:
            environment = ENVIRONMENTS[spec["family"]]
            if not (HERE / f".venv-{environment}/bin/python").is_file():
                run([HERE / "setup.sh", environment])
            flags = ["--views", str(spec["views"])]
            if spec["family"] == "smolvla":
                flags += ["--hoist-cross-kv"]
            if spec["family"] in ("smolvla", "xvla"):
                flags += ["--fps", str(spec["fps"])]
            for task in spec["tasks"]:
                flags += ["--task", task]
            run([HERE / "export.sh", checkpoint, out, *flags])
        finalize_bundle(out, spec)
        print(f"Bundle: {out}\nCopy it to the Jetson, then run: ./run.sh {out.name}")
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"export.sh: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
