"""The contract every backend implements, and the shared preprocessing.

Preprocessing is deliberately **shared** rather than each backend using its own
framework's helpers. The question under test is "what does the runtime cost", so
resize, tokenize and normalization are done identically for every backend and timed
separately, leaving the model execution as the only difference. Where a backend
cannot accept pre-processed input (an HTTP server that takes a JPEG), it says so via
`preprocess_owned = False` and its `preprocess_ms` is reported as part of `total_ms`.
"""

from __future__ import annotations

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


# ── bundle plumbing ──────────────────────────────────────────────────────────

def load_export_info(bundle: Path) -> dict:
    f = bundle / "export_info.json"
    return json.loads(f.read_text()) if f.exists() else {}


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
