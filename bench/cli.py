"""`python -m bench ...` — the whole harness from one entry point.

A run is a **model** crossed with a **backend**. `--model` pulls defaults out of the
registry (`bench/models.py`): where the weights live, where a split ONNX export lives,
which tokenizer, and the shapes needed to build an input. Any of it can be overridden
by passing the path directly, which is how a locally fine-tuned checkpoint is
benchmarked without touching the registry.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import models
from .backends.base import load_export_info
from .obs import make_obs
from .parity import load_results, parity_report
from .report import build_report
from .runner import collect_env, run_benchmark, write_result

DEFAULT_CACHE = str(Path.home() / ".cache" / "jetson-orin-nano-vla" / "trt")


# ── shared arguments ─────────────────────────────────────────────────────────

def _add_model(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None,
                   help=f"registry key ({', '.join(sorted(models.REGISTRY))}); "
                        f"supplies defaults that explicit paths override")
    p.add_argument("--family", choices=["smolvla", "xvla"], default=None,
                   help="required when benchmarking a path with no --model")
    p.add_argument("--task", default=None,
                   help="instruction (default: the model's, or the bundle's "
                        "export_info.json)")
    p.add_argument("--state-dim", type=int, default=None)
    p.add_argument("--action-dim", type=int, default=None,
                   help="action columns to compare; default is the model's full width")
    p.add_argument("--views", type=int, default=None,
                   help="number of REAL cameras to feed")
    p.add_argument("--cam-slots", type=int, default=None,
                   help="camera slots the export was built with. An unused slot still "
                        "occupies its tokens in the prefix, so PyTorch pads to this "
                        "count to stay comparable with the ONNX path")
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--noise-width", type=int, default=None,
                   help="padded action width the graph expects (SmolVLA 32, X-VLA 20); "
                        "only set this if the default is wrong for a custom export")
    p.add_argument("--fps", type=int, default=None,
                   help="dataset fps for the control-loop figures")


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
    p.add_argument("--monitor", choices=["auto", "tegrastats", "psutil"], default="auto")
    p.add_argument("--notes", default=None)


class Resolved:
    """Model defaults merged with whatever the command line overrode."""

    def __init__(self, args, bundle: Path | None = None):
        spec = models.get(args.model) if args.model else None
        info = load_export_info(bundle) if bundle else {}

        self.spec = spec
        self.info = info
        self.family = args.family or (spec.family if spec else None)
        if not self.family:
            sys.exit("cannot tell which model family this is: pass --model or --family.")

        self.task = args.task or info.get("task") or (spec.task if spec else None)
        if not self.task:
            sys.exit(
                "no task string. Pass --task, use --model, or point at a bundle whose "
                "export_info.json carries one. The policy conditions on the language "
                "embedding, so the wrong phrasing is a silently out-of-distribution run.")

        self.fps = args.fps or int(info.get("fps", 30))
        self.chunk_size = (args.chunk_size or info.get("chunk_size")
                           or (spec.chunk_size if spec else None) or 50)
        self.state_dim = (args.state_dim or (spec.state_dim if spec else None) or 6)
        # None means "compare every column the model emits" — stricter, and correct for
        # a base checkpoint whose real action width is embodiment-dependent.
        self.action_dim = args.action_dim or (spec.action_dim if spec else None)
        self.views = args.views or (spec.image_views if spec else 1)
        self.cam_slots = args.cam_slots or (spec.cam_slots if spec else None)
        self.tokenizer = spec.tokenizer if spec else None
        # Noise width is the model's PADDED action width, which is not the same as
        # how many columns we compare. SmolVLA emits 32 regardless of the robot's real
        # action dim (the rest is padding); X-VLA emits its real width. Getting this
        # from --action-dim would hand the graph a wrongly-shaped noise tensor.
        if args.noise_width:
            self.noise_width = args.noise_width
        elif self.family == "xvla":
            self.noise_width = (spec.action_dim if spec and spec.action_dim else 20)
        else:
            self.noise_width = int(info.get("max_action_dim", 32))

    def label(self, args, backend_name: str) -> str:
        if args.label:
            return args.label
        model = args.model or "local"
        return f"{backend_name}.{model}"


def _finish(args, backend, r: Resolved) -> int:
    obs = make_obs(args.obs, r.task, r.chunk_size, r.state_dim,
                   max_action_dim=r.noise_width, seed=args.seed, n_views=r.views)
    label = r.label(args, backend.name)
    result = run_benchmark(
        backend, obs, iters=args.iters, warmup=args.warmup, idle_s=args.idle_s,
        duration_s=args.duration_s, save_chunks=args.save_chunks, fps=r.fps,
        monitor_kind=None if args.monitor == "auto" else args.monitor,
        label=label, notes=args.notes)
    result["model"] = {"key": args.model, "family": r.family, "task": r.task,
                       "views": r.views, "cam_slots": r.cam_slots,
                       "state_dim": r.state_dim, "action_dim": r.action_dim}
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

    ckpt = args.checkpoint or (r.spec.torch_repo if r.spec else None)
    if not ckpt:
        sys.exit("no checkpoint: pass --checkpoint or --model.")
    if not Path(ckpt).exists():
        print(f"[torch] {ckpt} is not a local path — treating it as a Hugging Face id. "
              f"`python -m bench fetch --model {args.model}` downloads it first if you "
              f"would rather not do it inside the timed load.")

    if r.family == "xvla":
        from .backends.torch_xvla import TorchXVLABackend
        be = TorchXVLABackend(
            Path(ckpt), weights=args.weights, autocast=args.autocast,
            device=args.device, tokenizer=args.tokenizer or r.tokenizer
            or "facebook/bart-large", num_steps=args.num_steps,
            valid_views=r.views, compile_model=args.compile)
    else:
        from .backends.torch_smolvla import TorchSmolVLABackend
        be = TorchSmolVLABackend(
            Path(ckpt), bundle=bundle, weights=args.weights, autocast=args.autocast,
            device=args.device, action_dim=r.action_dim or 32,
            tokenizer_dir=Path(args.tokenizer or r.tokenizer) if (args.tokenizer or r.tokenizer) else None,
            compile_model=args.compile, patch_half_out=args.patch_half_out,
            cam_slots=r.cam_slots)
    return _finish(args, be, r)


def cmd_ort_split(args) -> int:
    bundle = Path(args.bundle) if args.bundle else None
    r = Resolved(args, bundle)
    if bundle is None:
        if r.spec and r.spec.split_repo:
            sys.exit(f"--bundle not given. Fetch the split export first:\n"
                     f"  python -m bench fetch --model {args.model} --what split\n"
                     f"then pass --bundle <that path>.")
        sys.exit("--bundle is required: the split ONNX export directory.")

    if r.family == "xvla":
        from .backends.ort_split_xvla import OrtSplitXVLABackend
        be = OrtSplitXVLABackend(
            bundle, cache_dir=args.cache_dir if args.cache_dir != DEFAULT_CACHE else None,
            precision=args.precision, num_steps=args.num_steps,
            tokenizer=args.tokenizer or r.tokenizer, seed=args.seed,
            valid_views=r.views)
    else:
        from .backends.ort_split import OrtSplitBackend
        be = OrtSplitBackend(
            bundle, cache_dir=args.cache_dir, precision=args.precision,
            num_steps=args.num_steps, action_dim=r.action_dim or 32,
            drop_cuda_ep=args.drop_cuda_ep, seed=args.seed,
            projectors=args.projectors, trt_opt_level=args.trt_opt_level,
            trt_workspace_mb=args.trt_workspace_mb,
            tokenizer=args.tokenizer or r.tokenizer, iobinding=args.iobinding)
    return _finish(args, be, r)


def cmd_ort_mono(args) -> int:
    from .backends.ort_mono import OrtMonoBackend
    bundle = Path(args.bundle) if args.bundle else None
    r = Resolved(args, bundle)
    be = OrtMonoBackend(
        Path(args.onnx), cache_dir=args.cache_dir, precision=args.precision,
        use_trt=not args.no_trt, bundle=bundle,
        tokenizer=args.tokenizer or r.tokenizer, action_dim=r.action_dim,
        drop_cuda_ep=args.drop_cuda_ep, trt_opt_level=args.trt_opt_level,
        trt_workspace_mb=args.trt_workspace_mb)
    return _finish(args, be, r)


def cmd_tether(args) -> int:
    from .backends.tether_http import TetherHttpBackend
    bundle = Path(args.bundle) if args.bundle else None
    r = Resolved(args, bundle)
    log = Path(args.serve_log) if args.serve_log else Path("results") / "tether-serve.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    be = TetherHttpBackend(
        export_dir=Path(args.export_dir) if args.export_dir else None,
        url=args.url, port=args.port, device=args.device, providers=args.providers,
        api_key=args.api_key, action_dim=r.action_dim or 32,
        startup_timeout_s=args.startup_timeout, payload_template=args.payload_template,
        extra_args=args.serve_arg or [], log_path=log)
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
        if s.notes:
            print(f"  note         : {s.notes}")
        print()
    return 0


def cmd_fetch(args) -> int:
    spec = models.get(args.model)
    cache = Path(args.dest).expanduser() if args.dest else Path.home() / "bundles"
    want = ["torch", "split"] if args.what == "both" else [args.what]
    for kind in want:
        repo = spec.torch_repo if kind == "torch" else spec.split_repo
        if not repo:
            print(f"!! {spec.key} has no {kind} artefact to fetch."
                  + (" Export it — see docs/03-backends.md." if kind == "split" else ""))
            continue
        print(f"fetching {kind}: {repo}")
        path = models.fetch(repo, subdir=f"{spec.key}-{kind}", cache=cache)
        print(f"  -> {path}")
    return 0


def cmd_report(args) -> int:
    md = build_report([Path(p) for p in args.paths], prefer_ref=args.reference)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


def cmd_parity(args) -> int:
    rep = parity_report(load_results([Path(p) for p in args.paths]), args.reference)
    print(json.dumps(rep, indent=2))
    bad = [c for c in rep.get("comparisons", [])
           if c.get("verdict") in ("FAIL", "SUSPECT", "SHAPE MISMATCH")]
    return 1 if bad else 0


def cmd_env(args) -> int:
    print(json.dumps(collect_env(), indent=2))
    return 0


def cmd_tether_probe(args) -> int:
    """Print the server's own /act schema, for when payload negotiation fails."""
    import requests
    base = args.url.rstrip("/")
    for path in ("/openapi.json", "/config", "/health"):
        try:
            r = requests.get(base + path, timeout=10)
            print(f"\n=== GET {path} -> {r.status_code} ===")
            body = r.json()
            if path == "/openapi.json":
                print(json.dumps(body.get("paths", {}).get("/act", {}), indent=2)[:4000])
                print("\n--- component schemas ---")
                print(json.dumps(body.get("components", {}).get("schemas", {}),
                                 indent=2)[:4000])
            else:
                print(json.dumps(body, indent=2)[:2000])
        except Exception as e:
            print(f"\n=== GET {path} -> {e}")
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
    p.add_argument("--weights", choices=["float32", "float16", "bfloat16"],
                   default="float32",
                   help="dtype the weights are held in (float32 is stock LeRobot)")
    p.add_argument("--autocast", choices=["off", "float16", "bfloat16"], default="off",
                   help="torch.autocast for the forward pass; float16 is the "
                        "mixed-precision path LeRobot itself trains with")
    p.add_argument("--patch-half-out", action="store_true",
                   help="SmolVLA + --weights float16: LeRobot 0.5.1 hardcodes an FP32 "
                        "cast in denoise_step. Disclosed in the run metadata.")
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    _add_model(p)
    _add_common(p)
    p.set_defaults(func=cmd_torch)

    p = sub.add_parser("ort-split", help="split ONNX export on ORT + TensorRT EP")
    p.add_argument("--bundle", default=None, help="the split export directory")
    p.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--num-steps", type=int, default=None,
                   help="denoise steps (default: from the export)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE,
                   help="TRT engine cache — keep it OFF /tmp, which clears at boot")
    # Default gpu, not cpu. Both are bit-identical in output (cosine 1.0000000, max abs
    # diff 0.000e+00), and gpu is 226 -> 169 ms while giving back 1.9 of six CPU cores.
    # Leaving the slow one as the default meant the headline number described a
    # configuration nobody would deploy. --projectors cpu still reproduces the stock
    # runtime for the A/B.
    p.add_argument("--projectors", choices=["cpu", "gpu"], default="gpu",
                   help="SmolVLA: where the four per-step projectors run. 'gpu' is the "
                        "default and is bit-identical to 'cpu', which reproduces the "
                        "stock runtime")
    p.add_argument("--trt-opt-level", type=int, default=None,
                   help="TensorRT builder optimization level (stock here is 2; TRT's "
                        "own default is 3). Clear the engine cache to force a rebuild")
    p.add_argument("--trt-workspace-mb", type=int, default=None,
                   help="per-tactic TRT scratch (stock here is 512)")
    # On by default for the same reason: bit-identical output, ~24%% off wall. The stock
    # loop re-feeds 32 KV tensors (7.2 MB) as numpy on every denoise step -- 72 MB of
    # host->device copies per inference for data that is constant after prefill.
    p.add_argument("--no-iobinding", dest="iobinding", action="store_false",
                   help="re-feed the KV cache as numpy every denoise step, as the stock "
                        "runtime does. Slower and bit-identical; for the A/B only")
    p.set_defaults(iobinding=True)
    p.add_argument("--drop-cuda-ep", action="store_true",
                   help="TRT_DROP_CUDA_EP=1: frees the 3 GiB CUDA arena for a tight build")
    p.add_argument("--tokenizer", default=None)
    _add_model(p)
    _add_common(p)
    p.set_defaults(func=cmd_ort_split)

    p = sub.add_parser("ort-mono",
                       help="MONOLITHIC ONNX on ORT — the split-vs-monolith A/B")
    p.add_argument("--onnx", required=True, help="the monolithic .onnx (+ .onnx.data)")
    p.add_argument("--bundle", default=None, help="for stats / tokenizer / task")
    p.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--no-trt", action="store_true",
                   help="CUDA EP only: no engine build, so the 8 GB build wall never "
                        "applies. This is the unoptimized-GPU baseline")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--trt-opt-level", type=int, default=None)
    p.add_argument("--trt-workspace-mb", type=int, default=None)
    p.add_argument("--drop-cuda-ep", action="store_true")
    p.add_argument("--tokenizer", default=None)
    _add_model(p)
    _add_common(p)
    p.set_defaults(func=cmd_ort_mono)

    p = sub.add_parser("tether", help="FastCrest Tether serve + /act")
    p.add_argument("--export-dir", default=None,
                   help="tether export dir to serve; omit to attach to a running server")
    p.add_argument("--url", default=None, help="attach to an already-running server")
    p.add_argument("--bundle", default=None, help="split bundle, for task/fps/chunk")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--providers", default=None,
                   help="comma-separated ORT providers, e.g. "
                        "TensorrtExecutionProvider,CUDAExecutionProvider")
    p.add_argument("--api-key", default=None)
    p.add_argument("--startup-timeout", type=float, default=900.0,
                   help="a first TensorRT build can take many minutes")
    p.add_argument("--payload-template", default=None,
                   help="JSON file using {{image_b64}} / {{instruction}} / {{state}}")
    p.add_argument("--serve-arg", action="append",
                   help="extra flag passed through to `tether serve` (repeatable)")
    p.add_argument("--serve-log", default=None)
    _add_model(p)
    _add_common(p)
    p.set_defaults(func=cmd_tether)

    p = sub.add_parser("models", help="what can be benchmarked")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("fetch", help="download a model's artefacts from Hugging Face")
    p.add_argument("--model", required=True)
    p.add_argument("--what", choices=["torch", "split", "both"], default="both")
    p.add_argument("--dest", default=None, help="default ~/bundles")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("report", help="run JSONs -> markdown tables")
    p.add_argument("paths", nargs="*", default=["results"])
    p.add_argument("--out", default=None)
    p.add_argument("--reference", default=None, help="label to use as parity reference")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("parity", help="compare saved action chunks across runs")
    p.add_argument("paths", nargs="*", default=["results"])
    p.add_argument("--reference", default=None)
    p.set_defaults(func=cmd_parity)

    p = sub.add_parser("env", help="print the machine fingerprint")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("tether-probe", help="print a running tether server's /act schema")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.set_defaults(func=cmd_tether_probe)

    p = sub.add_parser("selftest", help="check the monitors read real numbers")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--monitor", choices=["auto", "tegrastats", "psutil"], default="auto")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args(argv)
    return args.func(args)
