"""Build the engines for an exported bundle, check them against the stock policy, benchmark.

    ./run.sh <bundle> [--duration-s 300] [--baseline]

Engines are cached per bundle under ~/.cache/jetson-orin-nano-vla; the first run builds
them (5-15 min), later runs go straight to the benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bench import models

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "jetson-orin-nano-vla"

# AOT Triton stages per family: (stage, builder script, mode, extra flags). --score-fp32
# computes attention scores in FP32 where that measured faster and closer to stock.
AOT_STAGES = {
    "smolvla": [("vision", "build_vision.py", "triton-attention-flat", []),
                ("expert", "build_expert.py", "ffn", [])],
    "xvla": [("denoise", "build_denoise.py", "actions", ["--score-fp32"]),
             ("vision", "build_vision.py", "dwconv", [])],
    "evo1": [("output", "build_output.py", "gemv", []),
             ("step", "build_step.py", "attention", [])],
    "pi05": [("attention", "build_templates.py", "attention", ["--score-fp32"])],
    "groot": [("attention", "build.py", "all", [])],
}
# pi05-base's 50 action rows run faster on native TensorRT attention than on the
# kernel tuned for pi05-libero's 10 (118.5 vs 141.7 ms denoise).
NATIVE_ONLY = {"pi05-base"}


def detect_model(bundle: Path) -> str:
    if (bundle / "export_info.json").is_file():
        return "smolvla-base"
    info = json.loads((bundle / "bundle.json").read_text())
    if info.get("policy") == "pi05_libero":
        return "pi05-libero"
    if info.get("policy") == "pi05_lerobot":
        return "pi05-base"
    if info.get("model") == "evo1" or "random_action_head" in info:
        return "evo1-libero"
    if "deepstack" in info or "frames" in info:
        return "groot-n17-base"
    if "embodiment" in info:
        return "groot-n16-base"
    if "action_mode" in info or "domain_id" in info:
        return "xvla-base"
    raise ValueError(f"cannot tell which model {bundle} was exported from; pass --model")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inside(bundle: Path, name: str) -> Path:
    path = bundle / name
    if not path.resolve().is_relative_to(bundle.resolve()) or not path.is_file():
        raise ValueError(f"bundle is missing {name}; copy the whole export directory")
    return path


def validate_bundle(model: str, bundle: Path, optimized: bool) -> dict:
    """Refuse incomplete or altered bundles before spending minutes on engine builds."""
    meta = bundle / ("export_info.json" if model == "smolvla-base" else "bundle.json")
    info = json.loads(meta.read_text())
    if model == "smolvla-base":
        if not (info.get("fixture") or {}).get("file") or not info.get("embed_tokens"):
            raise ValueError("bundle lacks the stock-policy fixture; export it with ./export.sh")
        if optimized and ((info.get("kv") or {}).get("cross") != "expert-projected"
                          or info.get("n_cam_slots") != 2):
            raise ValueError("the SmolVLA AOT recipe is built for two-camera --hoist-cross-kv "
                             "exports (./export.sh smolvla); run other exports with --baseline")
    elif model in ("pi05-libero", "pi05-base"):
        if not info.get("complete"):
            raise ValueError("incomplete pi0.5 bundle")
        if optimized and model == "pi05-libero" and info.get("prefix_tokens") != 521:
            raise ValueError("the AOT attention recipe needs the compact 521-token prefix")
        if not list((bundle / "fixtures").glob("*.npz")):
            raise ValueError("pi0.5 bundle lacks stock-policy fixtures")
    else:
        fixture = (info.get("fixture") or {}).get("file")
        if not fixture:
            raise ValueError("bundle lacks the stock-policy fixture; export it with ./export.sh")
        inside(bundle, fixture)
        if model == "evo1-libero" and (info.get("random_action_head") or info.get("deployable") is False):
            raise ValueError("EVO1 bundle has a random action head, not trained weights")
    covered = set()
    for line in inside(bundle, "MANIFEST.sha256").read_text().splitlines():
        if not line.strip():
            continue
        sha, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if digest(inside(bundle, name)) != sha:
            raise ValueError(f"bundle checksum mismatch: {name}")
        covered.add(Path(name).as_posix())
    extra = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
             if p.is_file() and p.name != "MANIFEST.sha256"
             and not (p.name.startswith("_meta_") and p.suffix == ".json")} - covered
    if extra:
        raise ValueError(f"files not in the bundle manifest: {sorted(extra)[:5]}")
    return info


def cache_dir(model: str, bundle: Path, mode: str) -> Path:
    """One cache per bundle location, bundle content and engine-building source."""
    h = hashlib.sha256(str(bundle.resolve()).encode())
    h.update(digest(bundle / "MANIFEST.sha256").encode())
    family = models.get(model).family
    sources = [*sorted((ROOT / "bench/aot" / family).glob("*.py")), *sorted((ROOT / "bench/vendor").glob("*.py"))]
    for path in sources:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return CACHE_ROOT / f"{model}-{mode}-{h.hexdigest()[:12]}"


def plan(model: str, bundle: Path, cache: Path, optimized: bool, duration: float, out: Path):
    spec = models.get(model)
    py = sys.executable
    common = ["trt-split", "--model", model, "--bundle", str(bundle), "--chain", "graph",
              "--views", str(spec.image_views), "--accumulate", "fp32"]
    entry = [py, "-m", "bench"]
    if spec.family == "smolvla":
        # The saved Rust tokenizer, without importing Transformers.
        entry = [py, "-c", "from bench.aot.smolvla.runtime_memory import install_lean_tokenizer; "
                 "install_lean_tokenizer(); from bench.cli import main; raise SystemExit(main())"]
    base = cache / "baseline-cache"
    stages = [("TensorRT engines", base, [*entry, *common, "--cache-dir", str(base), "--iters", "3",
                                          "--warmup", "2", "--idle-s", "0",
                                          "--out", str(cache / "baseline.json")])]
    flags = []
    if optimized and model not in NATIVE_ONLY:
        specs = AOT_STAGES[spec.family]
        for i, (name, script, mode, extra) in enumerate(specs):
            dest = cache / ("optimized-cache" if i == len(specs) - 1 else f"{name}-cache")
            command = [py, f"bench/aot/{spec.family}/{script}", "--mode", mode, *extra,
                       "--bundle", str(bundle), "--base-cache", str(base), "--out", str(dest)]
            if spec.family != "pi05":
                command += ["--accumulate", "fp32"]
            stages.append((f"AOT Triton {name}", dest, command))
            base = dest
        entry = [py, f"bench/aot/{spec.family}/run_candidate.py"]
        if spec.family == "smolvla":
            flags = ["--lean-tokenizer", "--trim-host-heap"]
    bench = [*entry, *common, "--cache-dir", str(base), *flags, "--warmup", "10", "--idle-s", "3",
             "--duration-s", str(duration), "--label", f"{model}.{'aot' if optimized else 'trt'}",
             "--out", str(out)]
    return stages, bench


def execute(name: str, command: list[str], log: Path, cache: Path) -> None:
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TRITON_CACHE_DIR": str(cache / "triton-cache")}
    started = time.monotonic()
    print(f"  {name} ...", end="", flush=True)
    with log.open("w") as stream:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    print(f" {time.monotonic() - started:.0f} s" + ("" if code == 0 else " FAILED"), flush=True)
    if code:
        tail = log.read_text(errors="replace").splitlines()[-25:]
        print("\n".join("    " + line for line in tail), file=sys.stderr)
        raise SystemExit(f"{name} failed; full log: {log}")


def power_mode_note() -> None:
    try:
        mode = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return
    if "MAXN_SUPER" not in mode:
        print("note: the published numbers use MAXN_SUPER with jetson_clocks; this board is in "
              + (mode.splitlines()[0].split(":")[-1].strip() if mode else "an unknown mode"))


def summarize(result: Path) -> None:
    r = json.loads(result.read_text())
    lat = r.get("latency_ms", {})
    ram = (((r.get("system") or {}).get("windows") or {}).get("load") or {}).get("ram_used_mb", {})
    parity = (r.get("meta") or {}).get("fixture_parity") or {}
    chunk = (parity.get("reports") or {}).get("chunk") or {}
    print(f"\n{r.get('label')}: {r.get('status')}")
    print(f"  p50 {lat.get('p50')} ms   p95 {lat.get('p95')} ms   {lat.get('hz_mean')} Hz")
    if ram.get("mean"):
        print(f"  RAM in use {ram['mean'] / 1024:.2f} GB (whole board)")
    if parity:
        print(f"  parity vs stock policy: {parity.get('status')}"
              + (f" (cosine {chunk['cosine']:.6f}, max {chunk['max_pct_range']:.2f} % of range)"
                 if "cosine" in chunk else ""))
    print(f"  full result: {result}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="run.sh", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bundle", type=Path, help="directory written by ./export.sh")
    p.add_argument("--model", choices=list(models.REGISTRY), help="default: read from the bundle")
    p.add_argument("--duration-s", type=float, default=300, help="benchmark length (default 300)")
    p.add_argument("--baseline", action="store_true", help="plain TensorRT engines, no AOT Triton kernels")
    p.add_argument("--rebuild", action="store_true", help="discard this bundle's cached engines first")
    p.add_argument("--out", type=Path, help="result JSON (default: in the engine cache)")
    args = p.parse_args(argv)
    if args.duration_s <= 0:
        p.error("--duration-s must be positive")
    bundle = args.bundle.expanduser().resolve()
    if not bundle.is_dir():
        p.error(f"{bundle} is not a directory")

    model = args.model or detect_model(bundle)
    optimized = not args.baseline
    print(f"{model}: {bundle}")
    validate_bundle(model, bundle, optimized)
    power_mode_note()

    cache = cache_dir(model, bundle, "aot" if optimized else "trt")
    if args.rebuild and cache.exists():
        shutil.rmtree(cache)
    (cache / "logs").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (args.out.expanduser().resolve() if args.out else cache / "runs" / f"{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        p.error(f"{out} exists")

    stages, bench = plan(model, bundle, cache, optimized, args.duration_s, out)
    for name, dest, command in stages:
        done = dest.with_name(dest.name + ".done")
        if done.exists():
            print(f"  {name}: cached")
            continue
        if dest.exists():                      # an interrupted build
            shutil.rmtree(dest)
        execute(name, command, cache / "logs" / f"{dest.name}.log", cache)
        done.touch()
    execute(f"benchmark ({args.duration_s:.0f} s)", bench, cache / "logs" / f"benchmark-{stamp}.log", cache)
    summarize(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
