#!/usr/bin/env python3
"""Dump N frames from a LeRobot dataset episode to PNGs, for `--obs frames:DIR`.

Synthetic frames are enough for latency, power and CPU — the transformer does the
same work whatever the pixels are. They are NOT enough for anything where the action
values matter, because a procedural scene is out of distribution for a policy trained
on a sandbox. Point the benchmark at real frames when comparing what the runtimes
actually predict.

    python -m bench.tools.extract_frames \\
        --video ~/datasets/masi_digging_clean_ir/videos/.../cam1/episode_000000.mp4 \\
        --out frames/ --count 30 --stride 10

Reads the mp4 directly with OpenCV, so it needs neither lerobot nor torchcodec.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="episode .mp4 from the dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--stride", type=int, default=10, help="skip frames between saves")
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    import cv2

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {a.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)

    saved = idx = 0
    while saved < a.count:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % a.stride == 0:
            cv2.imwrite(str(out / f"frame_{saved:04d}.png"), frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"wrote {saved} frames to {out}")
    # A two-frame GOP means consecutive decoded frames on a static scene can be
    # near-identical; use --stride to get genuinely different observations.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
