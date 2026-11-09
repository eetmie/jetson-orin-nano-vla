"""Let the pinned LeRobot load a checkpoint saved by another LeRobot release or machine.

    python config_compat.py CHECKPOINT VIEW

Each case is handled in VIEW; the source checkpoint is never modified:
- config.json fields the pinned version's draccus parser rejects (newer releases add
  options such as inference-time RTC) are dropped.
- a tokenizer named by a local path that does not exist here (the trainer's machine)
  falls back to the policy class's default tokenizer, which its embedding was built for.
- an early LeRobot X-VLA conversion whose preprocessor lacks the image scaling and ImageNet
  steps gets them, as current LeRobot builds for the same config.
- a pre-processor checkpoint (normalization buffers inside model.safetensors, no
  policy_preprocessor.json) is converted by LeRobot's own migrate_policy_normalization.
VIEW holds symlinks to the unchanged files; it exists only if one of them applies.
It also reads the fps and task list of the dataset named in train_config.json (local
root, or metadata files from the Hub) so the bundle carries the trained prompts.
Prints {"dropped", "tokenizer", "migrated", "dataset"}.
"""
import dataclasses
import json
from pathlib import Path
import shutil
import subprocess
import sys


def policy_fields(config: dict) -> dict:
    from lerobot.configs.policies import PreTrainedConfig
    import lerobot.policies  # noqa: F401  (registers every policy config class)

    return {f.name: f for f in dataclasses.fields(PreTrainedConfig.get_choice_class(config["type"]))}


def missing_local(name) -> bool:
    return isinstance(name, str) and name.startswith(("/", "~", ".")) and not Path(name).expanduser().exists()


def dataset_meta(checkpoint: Path) -> dict | None:
    """fps and tasks (in task_index order) of the training dataset, or None."""
    train = checkpoint / "train_config.json"
    if not train.is_file():
        return None
    ds = json.loads(train.read_text()).get("dataset") or {}
    repo, root = ds.get("repo_id"), ds.get("root")
    try:
        if root and (Path(root) / "meta/info.json").is_file():
            def get(name):
                return Path(root) / "meta" / name if (Path(root) / "meta" / name).is_file() else None
        elif isinstance(repo, str) and repo.count("/") == 1 and not repo.startswith("local/"):
            from huggingface_hub import hf_hub_download

            def get(name):
                try:
                    return Path(hf_hub_download(repo, f"meta/{name}", repo_type="dataset", revision=ds.get("revision")))
                except Exception:
                    return None
        else:
            return None
        info = get("info.json")
        fps = json.loads(info.read_text()).get("fps") if info else None
        tasks = []
        if parquet := get("tasks.parquet"):  # LeRobot v3
            import pyarrow.parquet as pq
            table = pq.read_table(parquet).to_pydict()
            names = table.get("task") or table.get("__index_level_0__") or []
            order = table.get("task_index") or list(range(len(names)))
            tasks = [name for _, name in sorted(zip(order, names))]
        elif jsonl := get("tasks.jsonl"):  # LeRobot v2.1
            rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
            tasks = [r["task"] for r in sorted(rows, key=lambda r: r.get("task_index", 0))]
        return {"source": str(root or repo), "fps": fps, "tasks": tasks}
    except Exception as exc:
        print(f"NOTE: cannot read the training dataset metadata ({root or repo}): {exc}", file=sys.stderr)
        return None


def main(checkpoint: Path, view: Path) -> dict:
    config = json.loads((checkpoint / "config.json").read_text())
    fields = policy_fields(config)
    dropped = sorted(k for k in config if k != "type" and k not in fields)
    overrides = {}
    if dropped:
        config = {k: v for k, v in config.items() if k not in dropped}
        overrides["config.json"] = config
    tokenizer = None
    default = fields["tokenizer_name"].default if "tokenizer_name" in fields else config.get("vlm_model_name")
    if missing_local(config.get("tokenizer_name")):
        tokenizer = {"from": config["tokenizer_name"], "to": default}
        overrides["config.json"] = config = {**config, "tokenizer_name": default}
    pre_file = checkpoint / "policy_preprocessor.json"
    migrated = not pre_file.is_file()
    xvla_images = False
    if not migrated:
        pre = json.loads(pre_file.read_text())
        names = [step.get("registry_name") for step in pre.get("steps", [])]
        if config["type"] == "xvla" and "xvla_image_to_float" not in names and "tokenizer_processor" in names:
            # lerobot/xvla-{widowx,libero,folding} predate the image steps LeRobot added to xvla-base
            # (2025-12-03); without them the policy sees unnormalized pixels. Same order as xvla-base.
            at = names.index("tokenizer_processor") + 1
            pre["steps"][at:at] = [{"registry_name": "xvla_image_to_float", "config": {"image_keys": None, "validate_range": True}},
                                   {"registry_name": "xvla_imagenet_normalize", "config": {"image_keys": None}}]
            overrides["policy_preprocessor.json"] = pre
            xvla_images = True
        for step in pre.get("steps", []):
            name = (step.get("config") or {}).get("tokenizer_name")
            if missing_local(name):
                tokenizer = {"from": name, "to": default}
                step["config"]["tokenizer_name"] = default
                overrides["policy_preprocessor.json"] = pre
    source = checkpoint
    if overrides:
        source = view.with_name(view.name + ".fields") if migrated else view
        source.mkdir(parents=True)
        for item in checkpoint.iterdir():
            if item.name not in overrides:
                (source / item.name).symlink_to(item.resolve())
        for name, content in overrides.items():
            (source / name).write_text(json.dumps(content, indent=2) + "\n")
    if migrated:
        subprocess.run([sys.executable, "-m", "lerobot.processor.migrate_policy_normalization",
                        "--pretrained-path", str(source), "--output-dir", str(view)],
                       check=True, stdout=sys.stderr)
        if source != checkpoint:
            shutil.rmtree(source)
    return {"dropped": dropped, "tokenizer": tokenizer, "migrated": migrated, "xvla_image_steps_added": xvla_images,
            "dataset": dataset_meta(checkpoint)}


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]), Path(sys.argv[2]))))
