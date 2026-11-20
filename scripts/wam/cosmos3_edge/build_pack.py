"""Build the fixed-profile ONNX files sequentially with per-build memory telemetry."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def board_used():
    values = {
        k: int(v.split()[0])
        for k, v in (
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
    }
    return (values["MemTotal"] - values["MemAvailable"]) * 1024


def execute(command, log, platform):
    peak = 0
    with (
        open(log, "w") as out,
        subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT) as p,
    ):
        while p.poll() is None:
            value = (
                board_used()
                if platform == "jetson"
                else int(Path("/sys/fs/cgroup/memory.current").read_text())
            )
            peak = max(peak, value)
            if value > 7_300_000_000:
                p.terminate()
                raise RuntimeError("7.3 GB stop threshold reached")
            time.sleep(0.1)
        if p.returncode:
            raise RuntimeError(log.read_text()[-2500:])
    return peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("platform", choices=["spark", "jetson"])
    parser.add_argument("stages", nargs="*")
    parser.add_argument("--opt-level", default="1", help="TensorRT builder optimization level")
    parser.add_argument("--workspace", default="64", help="workspace pool, MiB")
    args = parser.parse_intermixed_args()
    root = args.root.resolve()
    report = root / (args.platform + "-memory.json")
    memory = json.loads(report.read_text()) if report.exists() else {}
    stages = args.stages or [
        "vae",
        "first",
        *[f"group{i:02d}" for i in range(14)],
        "last",
        "combine",
    ]
    exe = shutil.which("trtexec") or "/opt/tensorrt/bin/trtexec"
    for stage in stages:
        if stage == "run":
            cmd = [
                "python3",
                str(Path(__file__).with_name("run_full.py")),
                str(root),
                "--platform",
                args.platform,
                "--iters",
                "3",
            ]
        else:
            source = root / (stage + ".onnx")
            engine = root / (stage + ".engine")
            with source.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            if memory.get(stage, {}).get("sha256") == digest and engine.is_file():
                continue
            tmp = engine.with_suffix(".partial-engine")
            cmd = [
                exe,
                "--stronglyTyped",
                "--onnx=" + str(source),
                "--saveEngine=" + str(tmp),
                "--builderOptimizationLevel=" + args.opt_level,
                "--memPoolSize=workspace:" + args.workspace,
                "--skipInference",
            ]
        start = time.monotonic()
        peak = execute(
            cmd, root / (args.platform + "-" + stage + ".log"), args.platform
        )
        row = dict(peak_bytes=peak, elapsed_s=time.monotonic() - start)
        if stage != "run":
            tmp.replace(engine)
            row.update(sha256=digest, engine_bytes=engine.stat().st_size)
        memory[stage] = row
        report.write_text(json.dumps(memory, indent=2) + "\n")
        print(stage, row, flush=True)


if __name__ == "__main__":
    main()
