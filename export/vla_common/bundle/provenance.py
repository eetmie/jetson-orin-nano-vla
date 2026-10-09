# Vendored from the author's fine-tuning pipeline: vla-onnx/common/vla_common/bundle/provenance.py @ 3f7793d.
"""What produced this bundle — recorded so a result can be traced back to a stack."""

from __future__ import annotations

import subprocess
from pathlib import Path

# The packages whose version can change a traced graph or its numerics.
DEFAULT_PACKAGES = (
    "torch", "torchvision", "lerobot", "transformers",
    "onnx", "onnxruntime", "onnxscript", "numpy",
)


def package_version(name: str) -> str | None:
    import importlib.metadata as md

    try:
        return md.version(name)
    except Exception:
        return None


def package_versions(names: tuple[str, ...] = DEFAULT_PACKAGES) -> dict[str, str | None]:
    return {n: package_version(n) for n in names}


def git_sha(path: Path) -> str | None:
    """Short sha of the repo containing `path`, with -dirty when the tree is modified."""
    path = Path(path)
    repo = path if path.is_dir() else path.parent
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return None


def provenance(source: Path, *, opset: int | None = None, **extra) -> dict:
    """The provenance block written into bundle.json.

    `source` is the checkpoint or repo the bundle came from; anything else worth
    pinning (opset, domain_id, cam slots, chunk length) goes in as keyword args.
    """
    import datetime as _dt

    out = {
        "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": str(source),
        "git_sha": git_sha(Path(__file__)),
        "packages": package_versions(),
    }
    if opset is not None:
        out["opset"] = opset
    out.update(extra)
    return out
