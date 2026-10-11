"""Build and verify the optimized SmolVLA policy in fresh, explicit directories.

Uses a portable export/export.sh bundle; builds all engines on this Jetson.
No historical caches, experiment dates, or implicit home-directory bundles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAPHS = ("smolvlm_vision", "smolvlm_expert_prefill", "smolvlm_expert_decode",
          "state_projector", "action_in_projector", "action_out_projector",
          "time_in_projector", "time_out_projector")


def validate_bundle(bundle: Path):
    info = json.loads((bundle / "export_info.json").read_text())
    fixture = (info.get("fixture") or {}).get("file")
    embed = info.get("embed_tokens")
    if not fixture or not embed:
        raise ValueError("bundle needs stock-policy fixture and token embeddings; "
                         "export it with export/export.sh, not the old ORT-only HF bundle")
    if info.get("n_cam_slots") != 2:
        raise ValueError("optimized recipe requires a two-camera export (--views 2)")
    if (info.get("kv") or {}).get("cross") != "expert-projected":
        raise ValueError("selected optimized recipe requires a --hoist-cross-kv export")
    for name in (*[f"{g}.onnx" for g in GRAPHS], "stats.json", fixture, embed,
                 "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "MANIFEST.sha256"):
        path = bundle / name
        if not path.resolve().is_relative_to(bundle.resolve()):
            raise ValueError(f"bundle artifact outside bundle: {name}")
        if not path.is_file():
            raise ValueError(f"missing bundle artifact: {name}; copy the entire exported bundle")
    config = json.loads((bundle / "tokenizer/tokenizer_config.json").read_text())
    if config.get("tokenizer_class") != "GPT2Tokenizer":
        raise ValueError("optimized lean runtime requires the saved GPT2Tokenizer")
    # External ONNX weights and fixture/tokenizer files must travel with the graphs.
    covered = set()
    for line in (bundle / "MANIFEST.sha256").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        path = bundle / name.lstrip("*")
        if not path.resolve().is_relative_to(bundle.resolve()) or not path.is_file():
            raise ValueError(f"missing or invalid manifest artifact: {name}")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        if hasher.hexdigest() != digest:
            raise ValueError(f"bundle checksum mismatch: {name}")
        covered.add(path.relative_to(bundle).as_posix())
    artifacts = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
                 if p.is_file() and p.name != "MANIFEST.sha256"}
    if artifacts - covered:
        raise ValueError(f"bundle artifacts missing from manifest: {sorted(artifacts - covered)}")
    return info


def plan(args):
    work = args.work_dir
    common = ["trt-split", "--model", "smolvla-base", "--bundle", str(args.bundle),
              "--chain", "graph", "--views", "2", "--accumulate", args.accumulate,
              "--iters", "3"]
    # Run the baseline with the same lean tokenizer as the candidate. Each stage is
    # a fresh process; build-only imports cannot inflate the measured runtime RAM.
    baseline_code = ("from experiments.smolvla_triton.runtime_memory import install_lean_tokenizer; "
                     "install_lean_tokenizer(); from bench.cli import main; raise SystemExit(main())")
    return [
        ("runtime-doctor", [str(args.runtime_python), "-m", "bench.tools.doctor", "--role", "runtime",
                            "--out", str(work / "runtime-doctor.json")]),
        ("build-doctor", [str(args.build_python), "-m", "bench.tools.doctor", "--role", "build",
                          "--out", str(work / "build-doctor.json")]),
        ("baseline", [str(args.runtime_python), "-c", baseline_code, *common,
                      "--warmup", "2", "--idle-s", "0",
                      "--cache-dir", str(work / "baseline-cache"), "--label", "smolvla.baseline",
                      "--out", str(work / "baseline.json")]),
        ("vision", [str(args.build_python), "experiments/smolvla_triton/build_vision.py",
                    "--mode", "triton-attention-flat", "--bundle", str(args.bundle),
                    "--accumulate", args.accumulate, "--base-cache", str(work / "baseline-cache"),
                    "--out", str(work / "vision-cache")]),
        ("expert", [str(args.build_python), "experiments/smolvla_triton/build_expert.py",
                    "--mode", "ffn", "--bundle", str(args.bundle), "--accumulate", args.accumulate,
                    "--base-cache", str(work / "vision-cache"), "--out", str(work / "optimized-cache")]),
        ("optimized", [str(args.runtime_python), "experiments/smolvla_triton/run_candidate.py",
                       *common, "--cache-dir", str(work / "optimized-cache"),
                       "--lean-tokenizer", "--trim-host-heap", "--duration-s", str(args.duration_s),
                       "--warmup", "10", "--idle-s", "3", "--label", "smolvla.optimized",
                       "--out", str(work / "optimized.json")]),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="new directory; never reuses old engines")
    parser.add_argument("--runtime-python", type=Path, default=ROOT / ".venv-trt/bin/python")
    parser.add_argument("--build-python", type=Path, default=ROOT / ".venv-triton/bin/python")
    parser.add_argument("--accumulate", choices=("fp32", "auto"), default="fp32")
    parser.add_argument("--duration-s", type=float, default=300)
    parser.add_argument("--dry-run", action="store_true", help="print commands without imports, downloads or builds")
    args = parser.parse_args(argv)
    for name in ("bundle", "work_dir", "runtime_python", "build_python"):
        # Preserve venv interpreter symlinks (resolve() would run the system Python).
        setattr(args, name, Path(os.path.abspath(getattr(args, name).expanduser())))
    if args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    stages = plan(args)
    if args.dry_run:
        for name, command in stages:
            print(f"{name}: {shlex.join(command)}")
        return 0
    try:
        if args.work_dir.exists() or args.work_dir.resolve().is_relative_to(args.bundle.resolve()):
            raise ValueError("--work-dir must be a new directory outside the bundle")
        for executable in (args.runtime_python, args.build_python):
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ValueError(f"missing interpreter {executable}; run scripts/12_env_trt.sh and scripts/14_env_triton.sh")
        info = validate_bundle(args.bundle)
        args.work_dir.mkdir(parents=True)
        record = dict(bundle=str(args.bundle), work_dir=str(args.work_dir), accumulate=args.accumulate,
                      bundle_manifest_sha256=hashlib.sha256((args.bundle / "MANIFEST.sha256").read_bytes()).hexdigest(),
                      export_info=info, stages=[], status="running")
        record_path = args.work_dir / "setup.json"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        for name, command in stages:
            print(f"{name}: {shlex.join(command)}", flush=True)
            log = args.work_dir / f"{name}.log"
            with log.open("x") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONNOUSERSITE": "1",
                         "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                         "TRITON_CACHE_DIR": str(args.work_dir / "triton-cache")})
            record["stages"].append(dict(name=name, command=command, exit_code=result.returncode, log=str(log)))
            record["status"] = "failed" if result.returncode else "running"
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            if result.returncode:
                raise ValueError(f"{name} failed; see {log}. Use a new work directory after fixing it.")
        record["status"] = "pass"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Verified optimized cache: {args.work_dir / 'optimized-cache'}")
        return 0
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
