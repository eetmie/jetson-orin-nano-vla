# ─────────────────────────────────────────────────────────────────────────────
# Written for this repo (not vendored).
# ─────────────────────────────────────────────────────────────────────────────
"""Host image conversion helpers that keep each model's preprocessing bit-exact.

- normalize_table(expr): a model's uint8 -> float normalization, evaluated on every
  uint8 value per channel by the model's own expression, as a [1,256,3] float32 table;
  cv2.LUT through it gives the same HWC floats an order of magnitude faster than numpy.
- map_views(fn, views): per-camera preprocessing on a small thread pool (PIL, OpenCV
  and large numpy ops release the GIL); results keep camera order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

_POOL: ThreadPoolExecutor | None = None


def normalize_table(expr) -> np.ndarray:
    """expr maps a uint8 HxWx3 array to floats exactly as the model's code does."""
    values = np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(3, axis=2)
    return np.ascontiguousarray(np.asarray(expr(values)).astype(np.float32))


def lookup(image_u8: np.ndarray, table: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 -> HxWx3 float32 through a normalize_table."""
    import cv2

    return cv2.LUT(np.ascontiguousarray(image_u8), table)


def map_views(fn, views) -> list:
    views = list(views)
    if len(views) < 2:
        return [fn(v) for v in views]
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="views")
    return list(_POOL.map(fn, views))
