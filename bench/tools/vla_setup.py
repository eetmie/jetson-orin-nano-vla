"""Build and benchmark the README's six models with explicit bundles and fresh caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from bench import models
from bench.tools.smolvla_setup import validate_bundle as validate_smolvla

ROOT = Path(__file__).resolve().parents[2]
LISTED_MODELS = ("smolvla-base", "evo1-libero", "xvla-base", "groot-n17-base", "groot-n16-base", "pi05-libero")
# Runnable one at a time (--model M --bundle B), not part of --model all.
EXTRA_MODELS = ("pi05-base",)
PI05_POLICIES = {"pi05-libero": "pi05_libero", "pi05-base": "pi05_lerobot"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_path(bundle, name):
    p = bundle / name
    if not p.resolve().is_relative_to(bundle.resolve()) or not p.is_file():
        raise ValueError(f"missing/invalid bundle artifact {name}; copy the complete export")
    return p


def validate_bundle(model, bundle, optimized=True):
    if model == "smolvla-base" and optimized:
        return validate_smolvla(bundle)
    meta = bundle / ("export_info.json" if model == "smolvla-base" else "bundle.json")
    info = json.loads(meta.read_text())
    if model in PI05_POLICIES:
        if info.get("policy") != PI05_POLICIES[model] or not info.get("complete"):
            raise ValueError(f"{model} requires a complete {PI05_POLICIES[model]} pi0.5 bundle")
        if optimized and model == "pi05-libero" and info.get("prefix_tokens") != 521:
            raise ValueError("selected pi0.5 attention recipe requires the compact 521-token prefix")
        for name in ("norm_stats.json", "embedding_fp16.npy", "rope_cos_fp16.npy", "rope_sin_fp16.npy",
                     info.get("conditioning", "conditioning_10step_fp16.npy")):
            checked_path(bundle, name)
        for kind in {c["kind"] for c in info["components"]}:
            checked_path(bundle, f"templates/{kind}.onnx")
            checked_path(bundle, f"templates/{kind}.json")
        if models.get(model).task not in info.get("tasks", {}):
            raise ValueError("pi0.5 bundle lacks tokens for the benchmark's default prompt")
        if not list((bundle / "fixtures").glob("*.npz")):
            raise ValueError("pi0.5 bundle lacks stock-policy fixtures")
    else:
        fixture = (info.get("fixture") or {}).get("file")
        if not fixture:
            raise ValueError("native benchmarking requires a stock-policy fixture; run export/export.sh")
        checked_path(bundle, fixture)
        for graph in info.get("graphs", []):
            checked_path(bundle, graph if isinstance(graph, str) else graph["file"])
        if model == "evo1-libero" and (info.get("random_action_head") or info.get("deployable") is False):
            raise ValueError("evo1-libero requires trained weights, not the infrastructure bootstrap")
        expected = {"groot-n16-base": "nvidia/GR00T-N1.6-3B", "groot-n17-base": "nvidia/GR00T-N1.7-3B"}.get(model)
        if expected and info.get("model") != expected:
            raise ValueError(f"{model} requires a {expected} bundle")
    manifest = checked_path(bundle, "MANIFEST.sha256")
    lines = [line for line in manifest.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty bundle manifest")
    covered = set()
    for line in lines:
        sha, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if digest(checked_path(bundle, name)) != sha:
            raise ValueError(f"bundle checksum mismatch: {name}")
        covered.add(Path(name).as_posix())
    artifacts = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
                 if p.is_file() and p.name != "MANIFEST.sha256"
                 # Export scratch is excluded by write_manifest/ship_bundle.
                 and not (p.name.startswith("_meta_") and p.suffix == ".json")}
    if artifacts - covered:
        raise ValueError(f"bundle artifacts missing from manifest: {sorted(artifacts - covered)}")
    return info


def plan(model, bundle, work, runtime, builder, duration, accumulate="fp32", optimized=True):
    family = models.get(model).family
    common = ["trt-split", "--model", model, "--bundle", str(bundle), "--chain", "graph",
              "--views", str(models.get(model).image_views), "--accumulate", accumulate]
    # SmolVLA's lean tokenizer is also used for the baseline, avoiding a larger load.
    prefix = [str(runtime), "-m", "bench"]
    if family == "smolvla":
        prefix = [str(runtime), "-c", "from experiments.smolvla_triton.runtime_memory import install_lean_tokenizer; "
                  "install_lean_tokenizer(); from bench.cli import main; raise SystemExit(main())"]
    stages = [("runtime-doctor", [str(runtime), "-m", "bench.tools.doctor", "--role", "runtime", "--model", model,
                                  "--out", str(work / "runtime-doctor.json")])]
    # pi05-base runs native engines either way: the Triton action attention, tuned for
    # pi05-libero's 10 action rows, measured slower at its 50 (141.7 vs 118.5 ms denoise).
    use_plugins = optimized and model != "pi05-base"
    if use_plugins:
        stages.append(("build-doctor", [str(builder), "-m", "bench.tools.doctor", "--role", "build", "--model", model,
                                       "--out", str(work / "build-doctor.json")]))
        stages.append(("baseline", [*prefix, *common, "--cache-dir", str(work / "baseline-cache"),
                                    "--iters", "3", "--warmup", "2", "--idle-s", "0",
                                    "--out", str(work / "baseline.json")]))
        # (stage, builder, mode, extra flags). --score-fp32: attention scores in FP32 instead
        # of the graph's FP16 roundings; kept where it measured faster and closer to stock.
        specs = {"smolvla": [("vision", "build_vision.py", "triton-attention-flat", []), ("expert", "build_expert.py", "ffn", [])],
                 "xvla": [("denoise", "build_denoise.py", "actions", ["--score-fp32"]), ("vision", "build_vision.py", "dwconv", [])],
                 "evo1": [("output", "build_output.py", "gemv", []), ("step", "build_step.py", "attention", [])],
                 "pi05": [("attention", "build_templates.py", "attention", ["--score-fp32"])],
                 "groot": [("attention", "build.py", "all", [])]}[family]
        base = work / "baseline-cache"
        for i, (name, script, mode, extra) in enumerate(specs):
            dest = work / ("optimized-cache" if i == len(specs)-1 else f"{name}-cache")
            command = [str(builder), f"experiments/{family}_triton/{script}", "--mode", mode, *extra,
                       "--bundle", str(bundle), "--base-cache", str(base), "--out", str(dest)]
            if family != "pi05":
                command += ["--accumulate", accumulate]
            stages.append((name, command))
            base = dest
        prefix = [str(runtime), f"experiments/{family}_triton/run_candidate.py"]
        flags = ["--lean-tokenizer", "--trim-host-heap"] if family == "smolvla" else []
    else:
        base, flags = work / "baseline-cache", []
    stages.append(("benchmark", [*prefix, *common, "--cache-dir", str(base), *flags,
                                  "--warmup", "10", "--idle-s", "3", "--duration-s", str(duration),
                                  "--label", f"{model}.{'optimized' if optimized else 'baseline'}",
                                  "--out", str(work / "benchmark.json")]))
    return stages


def run_model(args, model, bundle, work):
    info = validate_bundle(model, bundle, args.mode == "optimized")
    stages = plan(model, bundle, work, args.runtime_python, args.build_python,
                  args.duration_s, args.accumulate, args.mode == "optimized")
    work.mkdir(parents=True, exist_ok=False)
    record = dict(model=model, bundle=str(bundle), bundle_manifest_sha256=digest(bundle / "MANIFEST.sha256"),
                  export_info=info, mode=args.mode, accumulate=args.accumulate, stages=[], status="running")
    record_path = work / "setup.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    for name, command in stages:
        print(f"{model}/{name}: {shlex.join(command)}", flush=True)
        log = work / f"{name}.log"
        with log.open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TRITON_CACHE_DIR": str(work / "triton-cache")})
        record["stages"].append(dict(name=name, command=command, exit_code=result.returncode, log=str(log)))
        record["status"] = "fail" if result.returncode else "running"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        if result.returncode:
            raise ValueError(f"{model}/{name} failed; see {log}")
    record["status"] = "pass"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return dict(model=model, status="pass", setup=str(record_path), result=str(work / "benchmark.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*LISTED_MODELS, *EXTRA_MODELS, "all"), required=True)
    parser.add_argument("--bundle", type=Path, help="one model's complete portable export")
    parser.add_argument("--bundles-dir", type=Path, help="all: directories named by model key")
    parser.add_argument("--work-dir", type=Path, required=True, help="new directory for logs, results and engines")
    parser.add_argument("--runtime-python", type=Path, default=ROOT / ".venv-trt/bin/python")
    parser.add_argument("--build-python", type=Path, default=ROOT / ".venv-triton/bin/python")
    parser.add_argument("--mode", choices=("optimized", "baseline"), default="optimized")
    parser.add_argument("--accumulate", choices=("fp32", "auto"), default="fp32")
    parser.add_argument("--duration-s", type=float, default=300)
    parser.add_argument("--keep-going", action="store_true", help="all: test remaining models after a failure")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    if args.model == "all" and (not args.bundles_dir or args.bundle):
        parser.error("--model all requires --bundles-dir and excludes --bundle")
    if args.model != "all" and (not args.bundle or args.bundles_dir):
        parser.error("one model requires --bundle and excludes --bundles-dir")
    for name in ("bundle", "bundles_dir", "work_dir", "runtime_python", "build_python"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(os.path.abspath(getattr(args, name).expanduser())))
    selected = LISTED_MODELS if args.model == "all" else (args.model,)
    jobs = [(model, args.bundles_dir / model if args.model == "all" else args.bundle,
             args.work_dir / model if args.model == "all" else args.work_dir) for model in selected]
    if args.dry_run:
        for model, bundle, work in jobs:
            for name, command in plan(model, bundle, work, args.runtime_python, args.build_python,
                                       args.duration_s, args.accumulate, args.mode == "optimized"):
                print(f"{model}/{name}: {shlex.join(command)}")
        return 0
    if args.work_dir.exists():
        parser.error("--work-dir exists; choose a new directory")
    for model, bundle, work in jobs:
        if work.resolve().is_relative_to(bundle.resolve()):
            parser.error("work directories must be outside the portable bundles")
    needed = [args.runtime_python]
    if args.mode == "optimized":
        needed.append(args.build_python)
    for executable in needed:
        if not executable.is_file() or not os.access(executable, os.X_OK):
            parser.error(f"missing {executable}; run scripts/12_env_trt.sh and scripts/14_env_triton.sh")
    if args.model == "all":
        args.work_dir.mkdir(parents=True)
    reports = []
    for model, bundle, work in jobs:
        try:
            reports.append(run_model(args, model, bundle, work))
        except (OSError, ValueError, KeyError) as exc:
            print(str(exc), file=sys.stderr)
            reports.append(dict(model=model, status="fail", error=str(exc)))
            if not args.keep_going:
                break
        if args.model == "all":
            (args.work_dir / "matrix.json").write_text(json.dumps(reports, indent=2) + "\n")
    if args.model == "all":
        (args.work_dir / "matrix.json").write_text(json.dumps(reports, indent=2) + "\n")
    return 0 if len(reports) == len(jobs) and all(r["status"] == "pass" for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
