"""`python -m bench trt-split ...`: one benchmark run of an exported bundle.

./run.sh drives this through the engine builds; call it directly to rerun a stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import models
from .backends.base import artifact_manifest, load_export_info
from .obs import make_obs
from .parity import attach_comparison_signature
from .runner import collect_env, run_benchmark, write_result

DEFAULT_CACHE = str(Path.home() / ".cache" / "jetson-orin-nano-vla" / "trt")


# ── shared arguments ─────────────────────────────────────────────────────────

def _add_model(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, choices=sorted(models.REGISTRY),
                   help="model profile to benchmark; explicit artifact paths may override "
                        "its download locations")
    p.add_argument("--task", default=None,
                   help="instruction (default: the model's)")
    p.add_argument("--views", type=int, default=None,
                   help="number of camera views (default: the base bundle contract)")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--label", help="name for this run (default: backend.model)")
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON (default results/<label>.json)")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--duration-s", type=float, default=None,
                   help="run for a wall-clock duration instead of a fixed count; "
                        "use this for the thermal-drift run (e.g. 300)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--idle-s", type=float, default=5.0,
                   help="baseline window with the model loaded and idle")
    p.add_argument("--obs", default="synthetic",
                   help="'synthetic' or 'frames:/path/to/dir'")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--save-chunks", type=int, default=8,
                   help="action chunks kept for the parity comparison")
    p.add_argument("--obs-ring-size", type=int, default=16,
                   help="bounded observations materialized before monitoring; duration "
                        "runs wrap the ring (it grows to include warmup + saved chunks)")
    p.add_argument("--monitor", choices=["auto", "tegrastats", "psutil"], default="auto")
    p.add_argument("--notes", default=None)


class Resolved:
    """Model defaults merged with whatever the command line overrode."""

    def __init__(self, args, bundle: Path | None = None):
        spec = models.get(args.model)
        info = load_export_info(bundle) if bundle else {}

        self.spec = spec
        self.info = info
        self.family = spec.family
        self.task = args.task or info.get("task") or spec.task
        if not self.task:
            sys.exit(
                "no task string. Pass --task or use the base model default. "
                "The policy conditions on the language embedding, so the wrong "
                "phrasing is a silently out-of-distribution run.")

        self.fps = info.get("fps") or 30
        self.chunk_size = info.get("chunk_size") or spec.chunk_size or 50
        self.state_dim = spec.state_dim or 6
        self.action_dim = spec.action_dim
        self.views = args.views if args.views is not None else spec.image_views
        self.cam_slots = spec.cam_slots
        self.tokenizer = spec.tokenizer
        self.noise_width = spec.action_dim or int(info.get("max_action_dim", 32))
        self.noise_distribution = spec.noise_distribution

    def label(self, args, backend_name: str) -> str:
        if args.label:
            return args.label
        model = args.model
        return f"{backend_name}.{model}"


def _finish(args, backend, r: Resolved) -> int:
    positive = {
        "iters": args.iters,
        "fps": r.fps,
        "chunk_size": r.chunk_size,
        "state_dim": r.state_dim,
        "views": r.views,
        "noise_width": r.noise_width,
        "obs_ring_size": args.obs_ring_size,
    }
    if r.action_dim is not None:
        positive["action_dim"] = r.action_dim
    if r.cam_slots is not None:
        positive["cam_slots"] = r.cam_slots
    if getattr(args, "num_steps", None) is not None:
        positive["num_steps"] = args.num_steps
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if args.warmup < 0 or args.idle_s < 0 or args.save_chunks < 0:
        raise ValueError("warmup, idle_s, and save_chunks must be non-negative")
    if args.duration_s is not None and args.duration_s <= 0:
        raise ValueError("duration_s must be positive when set")
    if r.cam_slots is not None and r.views > r.cam_slots:
        raise ValueError(
            f"views ({r.views}) exceeds the export camera slots ({r.cam_slots})")

    obs = make_obs(args.obs, r.task, r.chunk_size, r.state_dim,
                   max_action_dim=r.noise_width, seed=args.seed, n_views=r.views,
                   noise_distribution=r.noise_distribution)
    label = r.label(args, backend.name)
    result = run_benchmark(
        backend, obs, iters=args.iters, warmup=args.warmup, idle_s=args.idle_s,
        duration_s=args.duration_s, save_chunks=args.save_chunks, fps=r.fps,
        monitor_kind=None if args.monitor == "auto" else args.monitor,
        label=label, notes=args.notes, obs_ring_size=args.obs_ring_size)
    result["model"] = {"key": args.model, "family": r.family, "task": r.task,
                       "views": r.views, "cam_slots": r.cam_slots,
                       "state_dim": r.state_dim, "action_dim": r.action_dim}
    paths = backend.artifact_paths()
    if paths:
        result["artifacts"] = artifact_manifest(paths)
        result["validity"]["provenance"] = (
            "pass" if result["artifacts"]["complete"] else "fail")
    else:
        result["validity"]["provenance"] = "not_available"
    if result["status"] == "ok":
        attach_comparison_signature(result)
    out = args.out or Path("results") / f"{label}.json"
    write_result(result, out)
    lat = result.get("latency_ms", {})
    print(f"\n[{label}] {result['status']}  "
          f"p50={lat.get('p50')} ms  p95={lat.get('p95')} ms  "
          f"{lat.get('hz_mean')} Hz  -> {out}")
    if result["status"] != "ok":
        print(result.get("error"))
        return 1
    return 0


# ── subcommands ──────────────────────────────────────────────────────────────

def trt_backend(args, r: Resolved, bundle: Path):
    """The trt-split backend for a resolved model."""
    from .vendor import groot_trt
    groot_trt.ACCUMULATE = getattr(args, "accumulate", None) or groot_trt.ACCUMULATE
    # groot-n16-base keeps the directory its published run used.
    # FP32 accumulation (the default) gets its own directory, so engines built with
    # TensorRT's own choice (--accumulate auto) stay valid where they always were.
    suffix = "" if groot_trt.ACCUMULATE == "auto" or r.family == "pi05" else f"-{groot_trt.ACCUMULATE}acc"
    cache = args.cache_dir or str(Path(DEFAULT_CACHE).parent / (
        ("groot-trt" if args.model == "groot-n16-base" else f"{args.model}-trt") + suffix))
    if r.family == "groot":
        from .backends.trt_split_groot import TrtSplitGrootBackend
        return TrtSplitGrootBackend(bundle, cache_dir=cache,
                                    chain=getattr(args, "chain", "graph"))
    if r.family == "xvla":
        from .backends.trt_split_xvla import TrtSplitXVLABackend
        return TrtSplitXVLABackend(bundle, cache_dir=cache,
                                   chain=getattr(args, "chain", "graph"))
    if r.family == "smolvla":
        from .backends.trt_split_smolvla import TrtSplitSmolVLABackend
        return TrtSplitSmolVLABackend(bundle, cache_dir=cache, action_dim=r.action_dim or 32,
                                      chain=getattr(args, "chain", "graph"))
    if r.family == "evo1":
        from .backends.trt_split_evo1 import TrtSplitEvo1Backend
        return TrtSplitEvo1Backend(bundle, cache_dir=cache,
                                   chain=getattr(args, "chain", "graph"))
    if r.family == "pi05":
        from .backends.trt_split_pi05 import TrtSplitPi05Backend
        return TrtSplitPi05Backend(bundle, cache_dir=cache,
                                   chain=getattr(args, "chain", "graph"))
    sys.exit(f"trt-split does not serve {r.family}.")


def cmd_trt_split(args) -> int:
    bundle = Path(args.bundle) if args.bundle else None
    if bundle is None:
        sys.exit("--bundle is required: the split ONNX export directory (export/export.sh).")
    r = Resolved(args, bundle)
    return _finish(args, trt_backend(args, r, bundle), r)


def cmd_models(args) -> int:
    for s in models.REGISTRY.values():
        print(f"{s.key:16} {s.label:28} {s.checkpoint}")
    return 0


def cmd_env(args) -> int:
    print(json.dumps(collect_env(), indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench",
        description="VLA inference benchmarks on a Jetson Orin Nano 8 GB.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("trt-split", help="split ONNX export on the TensorRT runtime alone")
    p.add_argument("--bundle", default=None, help="the split export directory")
    p.add_argument("--cache-dir", default=None,
                   help="persistent TensorRT engine cache (default: one directory per model)")
    p.add_argument("--chain", choices=["graph", "device", "host"], default="graph",
                   help="graph (default): the device-resident chain replayed as a CUDA "
                        "graph; device: the same chain enqueued call by call; host: numpy "
                        "between engines, the reference the other two are checked against "
                        "at load")
    p.add_argument("--accumulate", choices=["fp32", "auto"], default="fp32",
                   help="fp32 (default): every FP16 MatMul accumulates in FP32 and FP32 "
                        "graphs run without TF32 (~1 %% slower, clearly more accurate); "
                        "auto: TensorRT picks, often FP16 accumulation. Engines are cached "
                        "per setting. pi0.5's graphs always accumulate in FP32.")
    _add_model(p)
    _add_common(p)
    p.set_defaults(func=cmd_trt_split)

    p = sub.add_parser("models", help="what can be benchmarked")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("env", help="print the machine fingerprint")
    p.set_defaults(func=cmd_env)

    args = ap.parse_args(argv)
    return args.func(args)
