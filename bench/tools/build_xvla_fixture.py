#!/usr/bin/env python3
"""Build an immutable aligned real-IR/state fixture from LeRobot episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/masi_digging_ir")
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint-tree-sha256", required=True)
    parser.add_argument("--video-backend", default="torchcodec")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"--out must be absent or empty: {args.out}")
    if len(set(args.episodes)) != len(args.episodes):
        parser.error("--episodes contains duplicates")
    args.out.mkdir(parents=True, exist_ok=True)
    images_dir = args.out / "images"
    images_dir.mkdir()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import cv2
    import lerobot
    import torch
    import torchcodec

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=args.episodes,
        video_backend=args.video_backend,
    )
    episode_values = np.asarray(
        [int(value) for value in dataset.hf_dataset["episode_index"]],
        dtype=np.int64,
    )
    info_path = args.dataset_root / "meta" / "info.json"
    stats_path = args.dataset_root / "meta" / "stats.json"
    tasks_path = args.dataset_root / "meta" / "tasks.parquet"
    info = json.loads(info_path.read_text())
    records = []

    for episode in args.episodes:
        positions = np.flatnonzero(episode_values == episode)
        if positions.size == 0:
            raise ValueError(f"episode {episode} has no records")
        position = int(positions[len(positions) // 2])
        sample = dataset[position]
        if sample["task"] != args.task:
            raise ValueError(
                f"episode {episode} task {sample['task']!r} != {args.task!r}")
        image_chw = sample["observation.images.cam1"].detach().cpu().numpy()
        image = np.rint(np.clip(image_chw, 0.0, 1.0) * 255.0).astype(
            np.uint8).transpose(1, 2, 0)
        image = np.ascontiguousarray(image)
        state = np.ascontiguousarray(
            sample["observation.state"].detach().cpu().numpy(), dtype=np.float32)
        action = np.ascontiguousarray(
            sample["action"].detach().cpu().numpy(), dtype=np.float32)
        frame_index = int(sample["frame_index"])
        dataset_index = int(sample["index"])
        relative = Path("images") / (
            f"episode_{episode:06d}_frame_{frame_index:06d}_cam1.png")
        if not cv2.imwrite(str(args.out / relative), image[:, :, ::-1]):
            raise RuntimeError(f"failed to write {relative}")
        decoded = cv2.imread(str(args.out / relative), cv2.IMREAD_COLOR)
        decoded = np.ascontiguousarray(decoded[:, :, ::-1])
        if not np.array_equal(decoded, image):
            raise RuntimeError(f"PNG round trip changed pixels for {relative}")
        records.append({
            "episode_index": episode,
            "frame_index": frame_index,
            "dataset_index": dataset_index,
            "timestamp": float(sample["timestamp"]),
            "task_index": int(sample["task_index"]),
            "state": state.tolist(),
            "state_sha256": array_sha256(state),
            "recorded_action": action.tolist(),
            "recorded_action_sha256": array_sha256(action),
            "images": [{
                "feature": "observation.images.cam1",
                "path": relative.as_posix(),
                "array_sha256": array_sha256(image),
                "file_sha256": sha256_file(args.out / relative),
            }],
        })

    metadata = {
        "schema_version": 1,
        "task": args.task,
        "views": 1,
        "image_hw": info["features"]["observation.images.cam1"]["shape"][:2],
        "state_dim": info["features"]["observation.state"]["shape"][-1],
        "action_dim": info["features"]["action"]["shape"][-1],
        "state_features": info["features"]["observation.state"]["names"],
        "action_features": info["features"]["action"]["names"],
        "checkpoint_tree_sha256": args.checkpoint_tree_sha256,
        "source_dataset": {
            "repo_id": args.repo_id,
            "root": str(args.dataset_root),
            "codebase_version": info.get("codebase_version"),
            "fps": info.get("fps"),
            "episodes": args.episodes,
            "info_sha256": sha256_file(info_path),
            "stats_sha256": sha256_file(stats_path),
            "tasks_sha256": sha256_file(tasks_path),
        },
        "loader": {
            "lerobot": lerobot.__version__,
            "torch": torch.__version__,
            "torchcodec": torchcodec.__version__,
            "video_backend": args.video_backend,
            "python_machine": platform.machine(),
        },
        "selection": "middle frame of each requested episode",
        "records": records,
    }
    output = args.out / "fixture.json"
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(records)} aligned records to {args.out}")


if __name__ == "__main__":
    main()
