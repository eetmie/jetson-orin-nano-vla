"""`python -m bench ...` — the whole harness from one entry point.

A run is a **model** crossed with a **backend**. `--model` pulls defaults out of the
registry (`bench/models.py`): where the weights live, where a split ONNX export lives,
which tokenizer, and the shapes needed to build an input. This repository registers
two public base checkpoints and one nondeployable EVO1 infrastructure bootstrap.
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
from .parity import (ResultLoadError, attach_comparison_signature, load_results,
                     parity_report)
from .report import build_report
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
                       "state_dim": r.state_dim, "action_dim": r.action_dim,
                       "deployable": r.spec.extras.get("deployable", True)}
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

def cmd_torch(args) -> int:
    bundle = Path(args.bundle) if args.bundle else None
    r = Resolved(args, bundle)

    if r.family == "evo1":
        sys.exit(
            "evo1-bootstrap has no Jetson PyTorch backend; its native LeRobot fixture "
            "is embedded in the split bundle for parity validation"
        )

    ckpt = args.checkpoint or r.spec.torch_repo
    if not ckpt:
        sys.exit("no checkpoint: pass --checkpoint.")
    if not Path(ckpt).exists():
        print(f"[torch] {ckpt} is not a local path — treating it as a Hugging Face id. "
              f"`python -m bench fetch --model {args.model}` downloads it first if you "
              f"would rather not do it inside the timed load.")

    if r.family == "xvla":
        from .backends.torch_xvla import TorchXVLABackend
        be = TorchXVLABackend(
            Path(ckpt), weights="float32", autocast="off", device="cuda",
            tokenizer=r.tokenizer or "facebook/bart-large", num_steps=None,
            valid_views=r.views, lang_len=r.spec.extras.get("lang_len"),
            compile_model=False)
    else:
        from .backends.torch_smolvla import TorchSmolVLABackend
        be = TorchSmolVLABackend(
            Path(ckpt), bundle=bundle, weights="float32", autocast="off",
            device="cuda", action_dim=r.action_dim or 32, tokenizer_dir=None,
            compile_model=False, patch_half_out=False,
            cam_slots=r.cam_slots, num_steps=None)
    return _finish(args, be, r)


def cmd_ort_split(args) -> int:
    bundle = Path(args.bundle) if args.bundle else None
    r = Resolved(args, bundle)
    if bundle is None:
        if r.spec.split_repo:
            sys.exit(f"--bundle not given. Fetch the split export first:\n"
                     f"  python -m bench fetch --model {args.model} --what split\n"
                     f"then pass --bundle <that path>.")
        sys.exit("--bundle is required: the split ONNX export directory.")

    if r.family == "evo1":
        from .backends.ort_split_evo1 import OrtSplitEvo1Backend
        be = OrtSplitEvo1Backend(
            bundle, cache_dir=args.cache_dir, precision="fp16",
            num_steps=args.num_steps)
    elif r.family == "xvla":
        from .backends.ort_split_xvla import OrtSplitXVLABackend
        be = OrtSplitXVLABackend(
            bundle, cache_dir=args.cache_dir if args.cache_dir != DEFAULT_CACHE else None,
            precision="fp16", num_steps=args.num_steps,
            tokenizer=r.tokenizer, seed=args.seed, valid_views=r.views)
    else:
        from .backends.ort_split import OrtSplitBackend
        be = OrtSplitBackend(
            bundle, cache_dir=args.cache_dir, precision="fp16",
            num_steps=args.num_steps, action_dim=r.action_dim or 32,
            drop_cuda_ep=False, seed=args.seed, projectors="gpu",
            trt_opt_level=None, trt_workspace_mb=None,
            tokenizer=None, iobinding=True)
    return _finish(args, be, r)


def cmd_models(args) -> int:
    print(models.describe())
    print()
    for s in models.REGISTRY.values():
        print(f"{s.key}: {s.label}")
        print(f"  task default : {s.task!r}")
        print(f"  chunk/steps  : {s.chunk_size} / {s.num_steps}")
        print(f"  state/action : {s.state_dim} / {s.action_dim or 'full padded width'}")
        print(f"  tokenizer    : {s.tokenizer}")
        print(f"  noise        : {s.noise_distribution}")
        if s.notes:
            print(f"  note         : {s.notes}")
        print()
    return 0


def cmd_fetch(args) -> int:
    spec = models.get(args.model)
    cache = Path(args.dest).expanduser() if args.dest else Path.home() / "bundles"
    want = ["torch", "split"] if args.what == "both" else [args.what]
    fetched = 0
    for kind in want:
        repo = spec.torch_repo if kind == "torch" else spec.split_repo
        if not repo:
            print(f"!! {spec.key} has no {kind} artefact to fetch."
                  + (" Export it — see docs/03-backends.md." if kind == "split" else ""))
            continue
        print(f"fetching {kind}: {repo}")
        path = models.fetch(repo, subdir=f"{spec.key}-{kind}", cache=cache)
        print(f"  -> {path}")
        fetched += 1
    return 0 if fetched else 2


def cmd_report(args) -> int:
    md = build_report([Path(p) for p in args.paths])
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


def cmd_parity(args) -> int:
    try:
        report = parity_report(
            load_results([Path(path) for path in args.paths]), args.reference)
    except (ResultLoadError, ValueError) as exc:
        report = {"error": str(exc)}
    print(json.dumps(report, indent=2))
    if report.get("error") or not report.get("comparisons"):
        return 2
    # Only numerical PASS is a successful parity gate. Distribution-only PLAUSIBLE
    # remains useful evidence but cannot certify cross-runtime equality.
    return 0 if all(c.get("verdict") == "PASS"
                    for c in report["comparisons"]) else 1


def cmd_env(args) -> int:
    print(json.dumps(collect_env(), indent=2))
    return 0


def cmd_selftest(args) -> int:
    """Prove the measurement plumbing works before trusting a number from it."""
    import time
    from .monitor import make_monitor
    mon = make_monitor(None if args.monitor == "auto" else args.monitor)
    print(f"sampler: {type(mon).__name__}")
    mon.start()
    time.sleep(0.5)
    with mon.window("idle"):
        time.sleep(args.seconds)
    with mon.window("load"):
        t0 = time.time()
        x = 0
        while time.time() - t0 < args.seconds:
            x += 1
    mon.stop()
    print(json.dumps(mon.summary(), indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench",
        description="VLA inference benchmarks on a Jetson Orin Nano 8 GB.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("torch", help="baseline: the LeRobot policy in PyTorch")
    p.add_argument("--checkpoint", default=None,
                   help="local pretrained_model dir, or an HF id (default: the model's)")
    p.add_argument("--bundle", default=None,
                   help="split export bundle, for stats.json / tokenizer / task")
    _add_model(p)
    _add_common(p)
    p.set_defaults(func=cmd_torch)

    p = sub.add_parser("ort-split", help="split ONNX export on ORT + TensorRT EP")
    p.add_argument("--bundle", default=None, help="the split export directory")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE,
                   help="persistent TensorRT engine cache")
    p.add_argument("--num-steps", type=int, default=None,
                   help="override the bundle flow/denoising steps")
    _add_model(p)
    _add_common(p)
    p.set_defaults(func=cmd_ort_split)

    p = sub.add_parser("models", help="what can be benchmarked")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("fetch", help="download a model's artefacts from Hugging Face")
    p.add_argument("--model", required=True, choices=sorted(models.REGISTRY))
    p.add_argument("--what", choices=["torch", "split", "both"], default="both")
    p.add_argument("--dest", default=None, help="default ~/bundles")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("report", help="run JSONs -> markdown tables")
    p.add_argument("paths", nargs="*", default=["results"])
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("parity", help="compare saved action chunks across runs")
    p.add_argument("paths", nargs="*", default=["results"])
    p.add_argument("--reference", required=True,
                   help="exact label of the reference run (required; parity fails closed)")
    p.set_defaults(func=cmd_parity)

    p = sub.add_parser("env", help="print the machine fingerprint")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("selftest", help="check the monitors read real numbers")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--monitor", choices=["auto", "tegrastats", "psutil"], default="auto")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args(argv)
    return args.func(args)
