"""The small normalization contract shared by host export and Jetson inference."""
from __future__ import annotations

import numpy as np


def transform(x, stats, *, mode="QUANTILES", implementation="lerobot", eps=1e-8, inverse=False):
    x = np.asarray(x, np.float32)
    if stats is None:
        return x
    def values(key):
        value = np.asarray(stats[key], None if implementation == "openpi" else np.float32).reshape(-1)
        if value.size < x.shape[-1]:
            raise ValueError(f"normalization {key} has {value.size} values, needs {x.shape[-1]}")
        return value[:x.shape[-1]]
    eps = float(eps) if implementation == "openpi" else np.float32(eps)
    if mode == "MEAN_STD":
        mean, std = values("mean"), values("std")
        return x * (std + eps) + mean if inverse else (x - mean) / (std + eps)
    if mode != "QUANTILES":
        raise ValueError(f"unsupported pi0.5 normalization {mode!r}")
    lo, hi = values("q01"), values("q99")
    if implementation == "openpi":
        scale = hi - lo + eps
        return (x + np.float32(1)) / np.float32(2) * scale + lo if inverse else (x - lo) / scale * np.float32(2) - np.float32(1)
    if implementation != "lerobot":
        raise ValueError(f"unsupported normalization implementation {implementation!r}")
    scale = hi - lo
    scale = np.where(scale == 0, eps, scale)
    return (x + np.float32(1)) * scale / np.float32(2) + lo if inverse else np.float32(2) * (x - lo) / scale - np.float32(1)
