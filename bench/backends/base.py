"""The contract every backend implements, and the shared preprocessing.

Preprocessing is deliberately **shared** rather than each backend using its own
framework's helpers. The question under test is "what does the runtime cost", so
resize, tokenize and normalization are done identically for every backend and timed
separately, leaving the model execution as the only difference. Where a backend
cannot accept pre-processed input (an HTTP server that takes a JPEG), it says so via
`preprocess_owned = False` and its `preprocess_ms` is reported as part of `total_ms`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..obs import Observation


@dataclass
class InferResult:
    chunk: np.ndarray                     # (chunk_size, action_dim), robot units
    timings_ms: dict[str, float] = field(default_factory=dict)  # must have "total"


class Backend:
    """One inference path under test."""

    name: str = "base"
    #: True when `infer` used the observation's injected noise, so its action chunk
    #: is directly comparable to another backend's. See bench/parity.py.
    noise_injected: bool = False
    #: True when the backend does its own preprocessing (so ours is not applied).
    preprocess_owned: bool = True

    def load(self) -> None: ...
    def meta(self) -> dict: return {}
    def infer(self, obs: Observation) -> InferResult: raise NotImplementedError
    def close(self) -> None: ...
    def pids(self) -> list[int]:
        """Processes to attribute CPU/RSS to. Default: just us."""
        import os
        return [os.getpid()]

    def artifact_paths(self) -> dict[str, Path]:
        """Named model inputs that must be hashed for a publishable result."""
        return {}


# ── bundle plumbing ──────────────────────────────────────────────────────────

def load_export_info(bundle: Path) -> dict:
    f = bundle / "export_info.json"
    return json.loads(f.read_text()) if f.exists() else {}


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_manifest(roots: dict[str, Path]) -> dict:
    """Hash every immutable artifact file, excluding generated engine/cache files."""
    output = {"roots": {}}
    for role, raw_root in sorted(roots.items()):
        root = Path(raw_root).expanduser()
        entry = {"path": str(root.resolve()), "files": []}
        if not root.exists():
            entry["error"] = "missing"
            output["roots"][role] = entry
            continue
        files = [root] if root.is_file() else sorted(
            file for file in root.rglob("*")
            if file.is_file()
            and file.suffix not in (".engine", ".pyc", ".log")
            and "trt_cache" not in file.parts
            and "__pycache__" not in file.parts
        )
        base = root.parent if root.is_file() else root
        for file in files:
            digest = file_sha256(file)
            entry["files"].append({
                "path": file.relative_to(base).as_posix(),
                "bytes": file.stat().st_size,
                "sha256": digest,
            })
        entry["manifest_sha256"] = hashlib.sha256(json.dumps(
            entry["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        output["roots"][role] = entry
    output["complete"] = bool(output["roots"]) and all(
        item.get("files") and not item.get("error")
        for item in output["roots"].values())
    output["manifest_sha256"] = hashlib.sha256(json.dumps(
        output["roots"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return output


def tree_sha256(path: Path) -> str | None:
    """Content identity for a tokenizer/config directory; independent of its location."""
    if not path.exists():
        return None
    if path.is_file():
        return file_sha256(path)
    files = sorted(file for file in path.rglob("*") if file.is_file())
    if not files:
        return None
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(bytes.fromhex(file_sha256(file)))
    return digest.hexdigest()


def load_stats(bundle: Path):
    """MEAN_STD stats from the bundle, as the split runtime reads them."""
    from ..vendor.smolvla_split import NormStats
    f = bundle / "stats.json"
    if not f.exists():
        return NormStats()
    stats = json.loads(f.read_text())
    # The bundle keys the image entry by camera; state/action keys are fixed.
    return NormStats.from_lerobot_stats(stats)


def bundle_camera_key(bundle: Path) -> str | None:
    f = bundle / "stats.json"
    if not f.exists():
        return None
    keys = [k for k in json.loads(f.read_text()) if k.startswith("observation.images.")]
    return keys[0] if keys else None
