"""Export π0.5 from a checksum-pinned converted checkpoint in the pinned container."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def link_or_copy(source, dest):
    """Share large weights when possible, including across Docker bind mounts."""
    try:
        os.link(source, dest)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, dest)
    return dest


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def finalize(bundle, initial, lock):
    import numpy as np
    info_path = bundle / "bundle.json"
    info = json.loads(info_path.read_text())
    references = json.loads((initial / "manifest.json").read_text())
    info["tasks"] = {}
    for fixture in references["fixtures"]:
        with np.load(initial / f"policy_fixture_{fixture['id']}.npz") as arrays:
            info["tasks"][fixture["prompt"]] = {
                "tokens": arrays["tokens"].reshape(-1).tolist(),
                "mask": arrays["token_mask"].reshape(-1).tolist(),
            }
    info["schema"] = "pi05-split/1"
    info["source_lock"] = lock
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    paths = sorted(p for p in bundle.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    (bundle / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(bundle)}\n" for p in paths))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    a.checkpoint, a.out, a.work_dir = (p.resolve() for p in (a.checkpoint, a.out, a.work_dir))
    if a.out.exists() or a.work_dir.exists():
        ap.error("--out and --work-dir must be new directories")
    if a.work_dir.is_relative_to(a.out) or a.out.is_relative_to(a.work_dir):
        ap.error("--out and --work-dir must be separate")
    lock = json.loads((HERE / "source-lock.json").read_text())
    from weight_digest import tensor_digest
    actual_weights = a.checkpoint / "model.safetensors"
    if tensor_digest(actual_weights, lock["unused_tensors"]) != lock["converted_active_tensors_sha256"]:
        ap.error("converted trained tensors differ from the pinned LIBERO model")
    lock["actual_converted_checkpoint_sha256"] = sha(actual_weights)
    initial, baseline = a.work_dir / "initial", a.work_dir / "baseline"
    initial.mkdir(parents=True)
    baseline.mkdir()
    from download import download as download_locked
    tokenizer_lock = json.loads((HERE / "tokenizer-lock.json").read_text())
    download_locked(tokenizer_lock, a.work_dir / "tokenizer")
    lock["tokenizer"] = tokenizer_lock
    env = {**os.environ, "PI05_CHECKPOINT": str(a.checkpoint), "PI05_INITIAL_ROOT": str(initial),
           "PI05_BASELINE_ROOT": str(baseline), "PYTHONNOUSERSITE": "1",
           "PI05_TOKENIZER": str(a.work_dir / "tokenizer" / "paligemma_tokenizer.model")}
    for script, stage in [("prepare_blocks.py", initial), ("prepare_conditioning.py", initial),
                          ("prepare_full.py", baseline), ("prepare_compact.py", a.out)]:
        if stage == a.out:
            a.out.mkdir(parents=True)
            shutil.copytree(baseline / "weights", a.out / "weights", copy_function=link_or_copy)
            shutil.copytree(baseline / "templates", a.out / "templates")
            link_or_copy(baseline / "embedding_fp16.npy", a.out / "embedding_fp16.npy")
        print(f"Export stage {script}: {stage}", flush=True)
        with (a.work_dir / f"{Path(script).stem}.log").open("x") as log:
            subprocess.run([sys.executable, str(HERE / script)], env={**env, "PI05_STAGE_ROOT": str(stage)},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    finalize(a.out, initial, lock)
    print(f"Complete compact bundle: {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
