"""`python -m bench ...` — the whole harness from one entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backends.base import load_export_info
from .obs import make_obs
from .report import build_report
from .parity import load_results, parity_report
from .runner import collect_env, run_benchmark, write_result

DEFAULT_CACHE = str(Path.home() / ".cache" / "jetson-orin-nano-vla" / "trt")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--label", help="name for this run (default: the backend name)")
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON (default results/<label>.json)")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--duration-s", type=float, default=None,
                   help="run for a wall-clock duration instead of a fixed count; "
                        "use this for the thermal-drift run (e.g. 180)")
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
    p.add_argument("--task", default=None,
                   help="instruction (default: read from the bundle's export_info.json)")
    p.add_argument("--fps", type=int, default=None,
                   help="dataset fps for the control-loop figures (default: bundle)")
    p.add_argument("--state-dim", type=int, default=3)
    p.add_argument("--action-dim", type=int, default=4)


def _resolve_task_fps(args, bundle: Path | None):
    info = load_export_info(bundle) if bundle else {}
    task = args.task or info.get("task")
    if not task:
        sys.exit("no task string: pass --task, or point at a bundle whose "
                 "export_info.json carries one. The policy conditions on the language "
                 "embedding, so the wrong phrasing is a silently out-of-distribution run.")
    return task, args.fps or int(info.get("fps", 30)), info


def _finish(args, backend, task, fps, chunk_size):
    obs = make_obs(args.obs, task, chunk_size, args.state_dim, seed=args.seed)
    label = args.label or backend.name
    result = run_benchmark(
        backend, obs, iters=args.iters, warmup=args.warmup, idle_s=args.idle_s,
        duration_s=args.duration_s, save_chunks=args.save_chunks, fps=fps,
        monitor_kind=None if args.monitor == "auto" else args.monitor,
        label=label, notes=args.notes)
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
    from .backends.torch_lerobot import TorchLeRobotBackend
    bundle = Path(args.bundle) if args.bundle else None
    task, fps, _ = _resolve_task_fps(args, bundle)
    be = TorchLeRobotBackend(Path(args.checkpoint), bundle=bundle,
                             weights=args.weights, autocast=args.autocast,
                             device=args.device, action_dim=args.action_dim,
                             compile_model=args.compile,
                             patch_half_out=args.patch_half_out)
    # chunk_size comes from the policy config, but the noise must be shaped before
    # the model is loaded — read it from the bundle, falling back to the config file.
    chunk = args.chunk_size or _chunk_from_checkpoint(Path(args.checkpoint), bundle)
    return _finish(args, be, task, fps, chunk)


def _chunk_from_checkpoint(ckpt: Path, bundle: Path | None) -> int:
    info = load_export_info(bundle) if bundle else {}
    if info.get("chunk_size"):
        return int(info["chunk_size"])
    cfg = ckpt / "config.json"
    if cfg.exists():
        return int(json.loads(cfg.read_text()).get("chunk_size", 50))
    return 50


def cmd_ort_split(args) -> int:
    from .backends.ort_split import OrtSplitBackend
    bundle = Path(args.bundle)
    task, fps, info = _resolve_task_fps(args, bundle)
    be = OrtSplitBackend(bundle, cache_dir=args.cache_dir, precision=args.precision,
                         num_steps=args.num_steps, action_dim=args.action_dim,
                         drop_cuda_ep=args.drop_cuda_ep, seed=args.seed)
    chunk = args.chunk_size or int(info.get("chunk_size", 50))
    return _finish(args, be, task, fps, chunk)


def cmd_tether(args) -> int:
    from .backends.tether_http import TetherHttpBackend
    bundle = Path(args.bundle) if args.bundle else None
    task, fps, info = _resolve_task_fps(args, bundle)
    log = Path(args.serve_log) if args.serve_log else Path("results") / "tether-serve.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    be = TetherHttpBackend(
        export_dir=Path(args.export_dir) if args.export_dir else None,
        url=args.url, port=args.port, device=args.device, providers=args.providers,
        api_key=args.api_key, action_dim=args.action_dim,
        startup_timeout_s=args.startup_timeout, payload_template=args.payload_template,
        extra_args=args.serve_arg or [], log_path=log)
    chunk = args.chunk_size or int(info.get("chunk_size", 50))
    return _finish(args, be, task, fps, chunk)


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
                act = body.get("paths", {}).get("/act", {})
                print(json.dumps(act, indent=2)[:4000])
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

    p = sub.add_parser("torch", help="baseline: LeRobot SmolVLAPolicy in PyTorch")
    p.add_argument("--checkpoint", required=True,
                   help="LeRobot pretrained_model dir (config.json + model.safetensors)")
    p.add_argument("--bundle", default=None,
                   help="split export bundle, for stats.json / tokenizer / task")
    p.add_argument("--weights", choices=["float32", "float16", "bfloat16"],
                   default="float32",
                   help="dtype the weights are held in (float32 is stock LeRobot)")
    p.add_argument("--autocast", choices=["off", "float16", "bfloat16"], default="off",
                   help="torch.autocast for the forward pass; float16 is the "
                        "mixed-precision path LeRobot itself trains with")
    p.add_argument("--patch-half-out", action="store_true",
                   help="needed with --weights float16: LeRobot 0.5.1 hardcodes an "
                        "FP32 cast in denoise_step. Disclosed in the run metadata.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    p.add_argument("--chunk-size", type=int, default=None)
    _add_common(p)
    p.set_defaults(func=cmd_torch)

    p = sub.add_parser("ort-split", help="split 9-graph ONNX on ORT + TensorRT EP")
    p.add_argument("--bundle", required=True, help="the split export bundle directory")
    p.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--num-steps", type=int, default=None,
                   help="denoise steps (default: from export_info.json)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE,
                   help="TRT engine cache — keep it OFF /tmp, which clears at boot")
    p.add_argument("--drop-cuda-ep", action="store_true",
                   help="TRT_DROP_CUDA_EP=1: frees the CUDA arena for a tight build")
    p.add_argument("--chunk-size", type=int, default=None)
    _add_common(p)
    p.set_defaults(func=cmd_ort_split)

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
    p.add_argument("--chunk-size", type=int, default=None)
    _add_common(p)
    p.set_defaults(func=cmd_tether)

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
