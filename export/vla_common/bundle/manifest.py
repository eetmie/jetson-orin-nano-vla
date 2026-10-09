# Vendored from the author's fine-tuning pipeline: vla-onnx/common/vla_common/bundle/manifest.py @ 3f7793d.
"""sha256 of every file in a bundle, so an rsync can be verified on arrival."""

from __future__ import annotations

import hashlib
from pathlib import Path

MANIFEST_NAME = "MANIFEST.sha256"


def write_manifest(out: Path, *, exclude_meta: bool = True) -> int:
    """Hash every shipped file in `out`, write MANIFEST.sha256, return the count.

    Written last and excludes itself. `sha256sum -c MANIFEST.sha256` on the Orin is
    the check that matters — the graphs travel as `.onnx` + `.onnx.data` pairs, and a
    truncated external-data file fails minutes later at engine-build time with an
    error that names neither the file nor the cause.

    `exclude_meta` drops `_meta_*.json`, the exporter's per-graph-family scratch used
    only to rebuild `bundle.json` on a partial re-export. `ship_bundle.sh` excludes
    them from the transfer, so a manifest that listed them would fail verification on
    the target for files that were never meant to travel. A manifest must describe
    the bundle AS SHIPPED. Pass False only for a bundle you are hashing in place.
    """
    out = Path(out)
    lines = []
    for f in sorted(out.rglob("*")):
        if not f.is_file() or f.name == MANIFEST_NAME:
            continue
        if exclude_meta and f.name.startswith("_meta_") and f.suffix == ".json":
            continue
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        lines.append(f"{h.hexdigest()}  {f.relative_to(out)}")
    (out / MANIFEST_NAME).write_text("\n".join(lines) + "\n")
    return len(lines)


def verify_manifest(out: Path) -> list[str]:
    """Re-hash the bundle and return the paths that disagree with the manifest.

    Empty list means the bundle is intact. Raises if the manifest is missing.
    """
    out = Path(out)
    manifest = out / MANIFEST_NAME
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found — bundle was never manifested")
    bad = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        f = out / rel
        if not f.is_file():
            bad.append(rel)
            continue
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != digest:
            bad.append(rel)
    return bad
