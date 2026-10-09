"""GR00T split bundles (N1.6, N1.7) on the TensorRT runtime (no ONNX Runtime)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from .base import Backend, InferResult


class TrtSplitGrootBackend(Backend):
    name = "trt-split-groot"
    noise_injected = True

    def __init__(self, bundle: Path, cache_dir: str, camera_fps: float = 30.0) -> None:
        self.bundle_dir = Path(bundle)
        self.cache_dir = cache_dir
        self.camera_fps = camera_fps
        self.bundle = None
        self.engines = None
        schema = json.loads((self.bundle_dir / "bundle.json").read_text()).get("schema", "")
        self.n17 = schema.startswith("groot-n1.7-split/")

    def load(self) -> None:
        from ..vendor.groot_trt import Engines, prebuild_engines

        if self.n17:
            from ..vendor.groot17_trt import Bundle, FrameHistory, validate_fixture
            self.bundle = Bundle(self.bundle_dir)
            # The history slot looks back -delta_indices[0] frames of the camera stream.
            lag = -self.bundle.b["video_delta_indices"][0] / self.camera_fps
            self.history = FrameHistory(lag)
            self.history_ages_ms: list[float] = []
        else:
            from ..vendor.groot_trt import Bundle, validate_fixture
            self.bundle = Bundle(self.bundle_dir)
        self.built = prebuild_engines(self.bundle, self.cache_dir)
        self.engines = Engines(self.bundle, self.cache_dir)
        self.fixture_parity = validate_fixture(self.bundle, self.engines)
        if self.fixture_parity["status"] != "PASS":
            raise ValueError(f"GR00T fixture parity failed: {self.fixture_parity}")

    def artifact_paths(self) -> dict[str, Path]:
        return {"bundle": self.bundle_dir}

    def meta(self) -> dict:
        import tensorrt as trt

        b = self.bundle.b
        m = {
            "backend": self.name,
            "family": "groot",
            "model": b.get("model", "nvidia/GR00T-N1.6-3B"),
            "bundle": str(self.bundle_dir),
            "precision": "mixed fp16 (strongly typed)",
            "num_steps": len(b["timesteps"]),
            "chunk_size": b["action_horizon"],
            "state_dim": b["state_dim"],
            "action_dim": b["action_dim"],
            "views": b["views"],
            "num_views": b["views"],
            "resize": b["image_hw"],
            "sequence_length": b["seq_len"],
            "prompt_tokens": b["prompt_tokens"],
            "embodiment": b["embodiment"],
            "embodiment_id": b["embodiment_id"],
            "n_graphs": len(self.bundle.names),
            "configured_provider_priority_per_graph": {
                n: "TensorrtExecutionProvider" for n in self.bundle.names},
            "runtime": "tensorrt python, no onnxruntime",
            "tensorrt": trt.__version__,
            "shared_scratch_mb": round(self.engines.scratch_bytes / 2**20, 1),
            "token_embedding": "fp16 .npy, memory-mapped on the CPU",
            "engine_cache": str(Path(self.cache_dir).expanduser()),
            "engines_built_this_load_s": self.built,
            "fixture_parity": self.fixture_parity,
        }
        if self.n17:
            m.update({
                "frames_per_view": b["frames"],
                "video_delta_indices": b["video_delta_indices"],
                "history": (f"each call encodes the {b['views']} current frames; the "
                            f"history slot reuses the encode of the newest frame at least "
                            f"{self.history.lag_s:.2f} s old ({self.camera_fps:g} fps "
                            "camera assumed), the current frame itself until one exists"),
                "state_used": b.get("state_used"), "action_used": b.get("action_used"),
            })
            if self.history_ages_ms:
                ages = np.array(self.history_ages_ms)
                m["history_age_ms"] = {"p50": round(float(np.median(ages)), 1),
                                       "min": round(float(ages.min()), 1),
                                       "max": round(float(ages.max()), 1), "calls": len(ages)}
        return m

    def _check(self, obs: Observation) -> None:
        b = self.bundle.b
        if len(obs.images) != b["views"]:
            raise ValueError(f"this GR00T bundle was exported for {b['views']} view(s), "
                             f"got {len(obs.images)}")
        if obs.task != b["task"]:
            raise ValueError(f"the bundle bakes the prompt {b['task']!r}; got {obs.task!r}. "
                             "Re-export with --task to change it.")

    def _state(self, obs: Observation) -> np.ndarray:
        # The stock processor min-max normalizes state and clips it to [-1, 1]; the
        # synthetic stream's raw N(0, 20^2) state is put in that same range.
        state = np.zeros((1, 1, self.bundle.b["state_dim"]), np.float32)
        state[0, 0, :obs.state.shape[0]] = np.clip(obs.state, -1.0, 1.0)
        return state

    def infer(self, obs: Observation) -> InferResult:
        self._check(obs)
        if self.n17:
            return self._infer17(obs)
        from ..vendor.groot_trt import infer

        t0 = time.perf_counter()
        pv = np.stack([self.bundle.preprocess(im) for im in obs.images])
        state = self._state(obs)
        pre = (time.perf_counter() - t0) * 1000
        t = {}
        actions = infer(self.bundle, self.engines.run, pv, state, obs.noise, timings=t)
        timings = {"total": pre + t["total"], "preprocess": pre,
                   "vision": t["vision"], "backbone": t["backbone"], "denoise": t["denoise"]}
        return InferResult(np.asarray(actions[0]), timings)

    def _infer17(self, obs: Observation) -> InferResult:
        from ..vendor.groot17_trt import encode_frames, infer

        t0 = time.perf_counter()
        pv = np.stack([self.bundle.preprocess(im) for im in obs.images])
        state = self._state(obs)
        t1 = time.perf_counter()
        now = encode_frames(self.bundle, self.engines.run, pv)
        self.history.push(t0, now)
        past, age = self.history.past(t0)
        t2 = time.perf_counter()
        t = {}
        actions = infer(self.bundle, self.engines.run, [past, now], state, obs.noise, timings=t)
        pre, vis = (t1 - t0) * 1000, (t2 - t1) * 1000
        self.history_ages_ms.append(age * 1000)
        timings = {"total": pre + vis + t["backbone"] + t["denoise"], "preprocess": pre,
                   "vision": vis, "backbone": t["backbone"], "denoise": t["denoise"]}
        return InferResult(np.asarray(actions[0]), timings)
