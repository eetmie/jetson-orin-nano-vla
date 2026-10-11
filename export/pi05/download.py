"""Download pinned public GCS generations and check their SHA-256."""
import argparse
import json
from pathlib import Path
import sys
import time
from urllib.parse import quote
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_common.http_download import RANGE_BYTES, verified, transfer as http_transfer


def transfer(url, item, partial, workers, *, resume=False):
    return http_transfer(url, item, partial, workers, open_url=urlopen, range_bytes=RANGE_BYTES, resume=resume)


def download(lock, target, *, workers=32, resume=False):
    if not 1 <= workers <= 64:
        raise ValueError("workers must be between 1 and 64")
    if target.is_symlink():
        raise ValueError("Checkpoint directory must not be a symlink")
    target.mkdir(parents=True, exist_ok=resume)
    for item in lock["objects"]:
        relative = Path(item["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid checkpoint path: {relative}")
        path = target / relative
        if not path.resolve().is_relative_to(target.resolve()) or path.is_symlink():
            raise ValueError(f"Unsafe checkpoint path: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not resume:
                raise FileExistsError(path)
            if not verified(path, item):
                raise ValueError(f"Existing checkpoint checksum mismatch: {relative}")
            print(f"Verified existing {relative}", flush=True)
            continue
        obj = quote(lock["prefix"] + item["file"], safe="")
        url = f"https://storage.googleapis.com/storage/v1/b/{lock['bucket']}/o/{obj}?alt=media&generation={item['generation']}"
        partial = path.with_name(path.name + ".partial")
        if partial.is_symlink():
            raise ValueError(f"Unsafe partial checkpoint path: {relative}")
        range_state = partial.with_name(partial.name + ".ranges.json")
        if range_state.is_symlink():
            raise ValueError(f"Unsafe checkpoint range state: {relative}")
        if resume and partial.exists() and not range_state.exists():
            partial.unlink()  # Legacy partials without range hashes restart.
        print(f"Download {relative} ({item['size']} bytes)", flush=True)
        transfer(url, item, partial, workers, resume=resume)
        if not verified(partial, item):
            raise ValueError(f"Checkpoint checksum mismatch: {relative}")
        partial.rename(path)
        range_state.unlink(missing_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="new checkpoint directory")
    ap.add_argument("--workers", type=int, default=32, help="parallel ranges for large objects (1 streams new files, resumes saved ranges serially)")
    ap.add_argument("--resume", action="store_true", help="verify completed files and saved ranges; restart legacy partials")
    a = ap.parse_args(argv)
    lock = json.loads((Path(__file__).parent / "checkpoint-lock.json").read_text())
    download(lock, a.out, workers=a.workers, resume=a.resume)


if __name__ == "__main__":
    main()
