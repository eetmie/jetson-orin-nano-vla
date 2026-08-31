#!/usr/bin/env python3
"""Run one non-headline X-VLA inference and save actual ORT node placement."""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.ort_profile import placement_verdict, summarize_profile
from bench.vendor.xvla_bundle_contract import verify_bundle
from bench.vendor.xvla_split_ort import XVLASplitPolicy, prebuild_engines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.num_steps is not None and args.num_steps <= 0:
        parser.error("--num-steps must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    prebuild_t0 = time.perf_counter()
    cache = prebuild_engines(args.bundle, str(args.cache_dir), args.precision)
    prebuild_s = time.perf_counter() - prebuild_t0
    bundle = verify_bundle(args.bundle, verify_manifest=False)
    load_t0 = time.perf_counter()
    policy = XVLASplitPolicy(
        args.bundle,
        cache_dir=str(args.cache_dir),
        precision=args.precision,
        num_denoising_steps=args.num_steps,
        seed=args.seed,
        profile_dir=raw_dir,
    )
    load_s = time.perf_counter() - load_t0
    configured = {
        name: session.get_providers()
        for name, session in policy._profile_sessions.items()
    }

    rng = np.random.default_rng(args.seed)
    images = [
        rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        for _ in range(policy.valid_views)
    ]
    state = rng.standard_normal(policy.state_dim).astype(np.float32)
    infer_t0 = time.perf_counter()
    action = policy.sample_actions(images, args.task, state)
    infer_ms = (time.perf_counter() - infer_t0) * 1000.0
    profile_paths = policy.end_profiling()

    per_graph = {}
    raw_files = {}
    for graph, profile_path in sorted(profile_paths.items()):
        per_graph[graph] = summarize_profile(profile_path)
        source = Path(profile_path)
        target = raw_dir / f"{graph}.json.gz"
        with source.open("rb") as input_stream, gzip.open(target, "wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
        source.unlink()
        raw_files[graph] = str(target.relative_to(args.out_dir))

    expected = [graph["name"] for graph in bundle["graphs"]]
    verdict = placement_verdict(per_graph, expected)
    result = {
        "status": "ok" if verdict["status"] == "pass" else "failed",
        "scope": "single synthetic validation inference; not a latency headline",
        "bundle": str(args.bundle),
        "bundle_checkpoint_tree_sha256": (bundle.get("checkpoint") or {}).get(
            "tree_sha256"),
        "cache": cache,
        "precision": args.precision,
        "task": args.task,
        "num_steps": policy.steps,
        "seed": args.seed,
        "load": {
            "cache_validation_s": round(prebuild_s, 3),
            "session_creation_s": round(load_s, 3),
        },
        "infer_ms_with_profiling": round(infer_ms, 3),
        "action_shape": list(action.shape),
        "configured_provider_priority": configured,
        "placement": verdict,
        "per_graph": per_graph,
        "raw_profiles": raw_files,
        "environment": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "kernel": platform.release(),
            "onnxruntime": ort.__version__,
        },
    }
    output = args.out_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "summary": str(output),
        "placement": verdict,
        "infer_ms_with_profiling": result["infer_ms_with_profiling"],
    }, indent=2))
    if verdict["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
