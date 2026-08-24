"""The observation stream every backend is fed — identical, deterministic, replayable.

A latency benchmark only needs *some* input, but a parity benchmark needs the *same*
input, and the two are worth measuring in one pass. So the observation source is
seeded and index-addressable: observation `i` is the same image, state, task string
and — critically — the same **flow-matching noise draw** for every backend.

The noise matters. SmolVLA's action head is a flow-matching ODE integrated from a
random `x_1`; two runs from an identical image disagree simply because they started
from different noise. Injecting the noise makes cross-backend action chunks directly
comparable, which turns parity from a separate script into a free by-product of the
benchmark run. Backends that cannot accept injected noise (an HTTP server, say)
record `noise_injected: false` and are compared statistically instead — see
`bench/parity.py`.

Sources
-------
`synthetic`   deterministic procedural scene. No camera, no dataset, no files.
              Use for latency/power/CPU; the pixels are not real so the *actions*
              are meaningless (but still comparable across backends).
`frames:DIR`  cycle real .png/.jpg frames. Use for anything where the action values
              matter. `tools/extract_frames.py` dumps these from a LeRobot dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_H, DEFAULT_W = 480, 640


@dataclass
class Observation:
    index: int
    image: np.ndarray          # uint8 HxWx3, the camera frame as the robot sees it
    state: np.ndarray          # float32 (n_joints,), raw units (degrees), NOT normalized
    task: str
    noise: np.ndarray          # float32 (1, chunk_size, max_action_dim)


class ObsSource:
    def __init__(self, task: str, chunk_size: int, state_dim: int,
                 max_action_dim: int = 32, seed: int = 1234,
                 state_scale: float = 20.0):
        self.task = task
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.max_action_dim = max_action_dim
        self.seed = seed
        self.state_scale = state_scale

    def _rng(self, index: int) -> np.random.Generator:
        # Per-index seeding, so obs[i] does not depend on how many were drawn before
        # it. Backends can therefore be run in any order, or resumed, and still line up.
        return np.random.default_rng([self.seed, index])

    def _state(self, index: int) -> np.ndarray:
        r = self._rng(index)
        return (r.standard_normal(self.state_dim) * self.state_scale).astype(np.float32)

    def _noise(self, index: int) -> np.ndarray:
        # Own generator so the noise draw is unaffected by image/state generation.
        r = np.random.default_rng([self.seed, index, 991])
        return r.standard_normal(
            (1, self.chunk_size, self.max_action_dim)).astype(np.float32)

    def __getitem__(self, index: int) -> Observation:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class SyntheticObs(ObsSource):
    """A procedural scene: gradient ground, a moving blob, seeded speckle.

    Deliberately not pure noise — a flat-random image is an unrealistic activation
    pattern for the SigLIP tower, and while that does not change latency it makes the
    output chunks less meaningful to eyeball. This is cheap and reproducible.
    """

    def __init__(self, *a, h: int = DEFAULT_H, w: int = DEFAULT_W, **kw):
        super().__init__(*a, **kw)
        self.h, self.w = h, w
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self._yy, self._xx = yy / h, xx / w

    def __getitem__(self, index: int) -> Observation:
        r = self._rng(index)
        phase = (index % 60) / 60.0
        img = np.zeros((self.h, self.w, 3), dtype=np.float32)
        img[..., 0] = 60 + 120 * self._yy                       # ground gradient
        img[..., 1] = 50 + 100 * (1.0 - self._yy)
        img[..., 2] = 70 + 60 * self._xx
        cx, cy = 0.2 + 0.6 * phase, 0.35 + 0.2 * np.sin(phase * 6.28)
        blob = np.exp(-(((self._xx - cx) ** 2 + (self._yy - cy) ** 2) / 0.01))
        img += (blob * 110.0)[..., None]
        img += r.normal(0, 6.0, img.shape)                      # sensor speckle
        return Observation(index, np.clip(img, 0, 255).astype(np.uint8),
                           self._state(index), self.task, self._noise(index))

    def describe(self) -> dict:
        return {"kind": "synthetic", "hw": [self.h, self.w], "seed": self.seed,
                "note": "procedural scene; action VALUES are not physically meaningful"}


class FrameDirObs(ObsSource):
    """Cycle real frames from a directory (sorted). State and noise stay seeded."""

    def __init__(self, directory: str | Path, *a, **kw):
        super().__init__(*a, **kw)
        self.dir = Path(directory)
        self.files = sorted(p for p in self.dir.iterdir()
                            if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not self.files:
            raise FileNotFoundError(f"no .png/.jpg frames in {self.dir}")
        self._cache: dict[int, np.ndarray] = {}

    def _load(self, i: int) -> np.ndarray:
        if i not in self._cache:
            import cv2
            img = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"could not decode {self.files[i]}")
            # cv2 gives BGR; the recorded pipeline feeds RGB to the policy.
            self._cache[i] = np.ascontiguousarray(img[:, :, ::-1])
        return self._cache[i]

    def __getitem__(self, index: int) -> Observation:
        return Observation(index, self._load(index % len(self.files)),
                           self._state(index), self.task, self._noise(index))

    def describe(self) -> dict:
        return {"kind": "frames", "dir": str(self.dir), "n_files": len(self.files),
                "seed": self.seed}


def make_obs(spec: str, task: str, chunk_size: int, state_dim: int,
             max_action_dim: int = 32, seed: int = 1234) -> ObsSource:
    """`synthetic` or `frames:/path/to/dir`."""
    if spec == "synthetic":
        return SyntheticObs(task, chunk_size, state_dim, max_action_dim, seed)
    if spec.startswith("frames:"):
        return FrameDirObs(spec.split(":", 1)[1], task, chunk_size, state_dim,
                           max_action_dim, seed)
    raise ValueError(f"unknown obs spec {spec!r} (want 'synthetic' or 'frames:DIR')")
