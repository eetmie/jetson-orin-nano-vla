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
`fixture:DIR` immutable aligned images + raw state from recorded episodes. Use for
              hardened real-input parity; `fixture.json` carries source identities.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_H, DEFAULT_W = 480, 640


@dataclass
class Observation:
    """One timestep, as every backend receives it.

    `images` is a list because the families differ: SmolVLA's reference export carries
    two camera slots, X-VLA declares three views. Give each backend only the *real*
    cameras — padded views are zeroed or filled with the convention image by the
    runtime itself and never need a forward pass, so handing over a fake frame would
    buy a vision-tower pass that the real deployment does not pay for.
    """

    index: int
    images: list[np.ndarray]   # uint8 HxWx3 each, in camera order
    state: np.ndarray          # float32 (state_dim,), raw units, NOT normalized
    task: str
    noise: np.ndarray          # float32 (1, chunk_size, action_width)

    @property
    def image(self) -> np.ndarray:
        """The first camera — what a single-view policy sees."""
        return self.images[0]


class ObsSource:
    def __init__(self, task: str, chunk_size: int, state_dim: int,
                 max_action_dim: int = 32, seed: int = 1234,
                 state_scale: float = 20.0, n_views: int = 1):
        self.task = task
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.max_action_dim = max_action_dim
        self.seed = seed
        self.state_scale = state_scale
        self.n_views = n_views

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

    def _view(self, index: int, view: int) -> np.ndarray:
        r = np.random.default_rng([self.seed, index, view])
        phase = ((index + 17 * view) % 60) / 60.0
        img = np.zeros((self.h, self.w, 3), dtype=np.float32)
        img[..., 0] = 60 + 120 * self._yy                       # ground gradient
        img[..., 1] = 50 + 100 * (1.0 - self._yy)
        img[..., 2] = 70 + 60 * self._xx
        cx, cy = 0.2 + 0.6 * phase, 0.35 + 0.2 * np.sin(phase * 6.28)
        blob = np.exp(-(((self._xx - cx) ** 2 + (self._yy - cy) ** 2) / 0.01))
        img += (blob * 110.0)[..., None]
        img += r.normal(0, 6.0, img.shape)                      # sensor speckle
        return np.clip(img, 0, 255).astype(np.uint8)

    def __getitem__(self, index: int) -> Observation:
        return Observation(index, [self._view(index, v) for v in range(self.n_views)],
                           self._state(index), self.task, self._noise(index))

    def describe(self) -> dict:
        return {"kind": "synthetic", "hw": [self.h, self.w], "views": self.n_views,
                "seed": self.seed,
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
        # Multi-view from one directory: consecutive files stand in for cameras, so a
        # second view is a genuinely different image rather than a copy.
        imgs = [self._load((index * self.n_views + v) % len(self.files))
                for v in range(self.n_views)]
        return Observation(index, imgs, self._state(index), self.task,
                           self._noise(index))

    def describe(self) -> dict:
        return {"kind": "frames", "dir": str(self.dir), "n_files": len(self.files),
                "views": self.n_views, "seed": self.seed}


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


class FixtureObs(ObsSource):
    """Replay immutable aligned recorded images and raw robot state."""

    def __init__(self, directory: str | Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dir = Path(directory)
        metadata_path = self.dir / "fixture.json"
        try:
            self.metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid observation fixture {metadata_path}: {exc}") from exc
        if self.metadata.get("schema_version") != 1:
            raise ValueError("observation fixture schema_version must be 1")
        if self.metadata.get("task") != self.task:
            raise ValueError(
                f"fixture task {self.metadata.get('task')!r} does not match "
                f"requested task {self.task!r}")
        if int(self.metadata.get("state_dim") or 0) != self.state_dim:
            raise ValueError(
                f"fixture state_dim={self.metadata.get('state_dim')} does not match "
                f"requested state_dim={self.state_dim}")
        if int(self.metadata.get("views") or 0) != self.n_views:
            raise ValueError(
                f"fixture views={self.metadata.get('views')} does not match "
                f"requested views={self.n_views}")
        self.records = self.metadata.get("records") or []
        if not self.records:
            raise ValueError("observation fixture contains no records")
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self.fixture_sha256 = _tree_sha256(self.dir)

    def _load(self, record_index: int, view: int) -> np.ndarray:
        key = (record_index, view)
        if key not in self._cache:
            relative = Path(self.records[record_index]["images"][view]["path"])
            path = (self.dir / relative).resolve()
            try:
                path.relative_to(self.dir.resolve())
            except ValueError as exc:
                raise ValueError(f"fixture image escapes its root: {relative}") from exc
            import cv2
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"could not decode fixture image {path}")
            image = np.ascontiguousarray(image[:, :, ::-1])
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            expected = self.records[record_index]["images"][view]["array_sha256"]
            if digest != expected:
                raise ValueError(f"fixture image array hash mismatch: {path}")
            self._cache[key] = image
        return self._cache[key]

    def __getitem__(self, index: int) -> Observation:
        record_index = index % len(self.records)
        record = self.records[record_index]
        state = np.asarray(record["state"], dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(
                f"fixture record state shape {state.shape} != {(self.state_dim,)}")
        state_digest = hashlib.sha256(state.tobytes()).hexdigest()
        if state_digest != record["state_sha256"]:
            raise ValueError(f"fixture state hash mismatch in record {record_index}")
        images = [self._load(record_index, view) for view in range(self.n_views)]
        return Observation(index, images, state, self.task, self._noise(index))

    def describe(self) -> dict:
        return {
            "kind": "fixture",
            "dir": str(self.dir),
            "fixture_sha256": self.fixture_sha256,
            "n_records": len(self.records),
            "hw": self.metadata.get("image_hw"),
            "views": self.n_views,
            "state_dim": self.state_dim,
            "seed": self.seed,
            "source_dataset": self.metadata.get("source_dataset"),
            "record_ids": [
                {
                    "episode_index": record.get("episode_index"),
                    "frame_index": record.get("frame_index"),
                    "dataset_index": record.get("dataset_index"),
                }
                for record in self.records
            ],
        }


def make_obs(spec: str, task: str, chunk_size: int, state_dim: int,
             max_action_dim: int = 32, seed: int = 1234,
             n_views: int = 1) -> ObsSource:
    """`synthetic` or `frames:/path/to/dir`."""
    if spec == "synthetic":
        return SyntheticObs(task, chunk_size, state_dim, max_action_dim, seed,
                            n_views=n_views)
    if spec.startswith("frames:"):
        return FrameDirObs(spec.split(":", 1)[1], task, chunk_size, state_dim,
                           max_action_dim, seed, n_views=n_views)
    if spec.startswith("fixture:"):
        return FixtureObs(spec.split(":", 1)[1], task, chunk_size, state_dim,
                          max_action_dim, seed, n_views=n_views)
    raise ValueError(
        f"unknown obs spec {spec!r} "
        "(want 'synthetic', 'frames:DIR', or 'fixture:DIR')")
