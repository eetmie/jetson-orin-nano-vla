"""GR00T split bundles (N1.6, N1.7) on the TensorRT runtime (no ONNX Runtime)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from ..vendor.imaging import map_views
from .base import Backend, InferResult


class TrtSplitGrootBackend(Backend):
    name = "trt-split-groot"
    noise_injected = True

    def __init__(self, bundle: Path, cache_dir: str, camera_fps: float = 30.0,
                 chain: str = "host") -> None:
        self.bundle_dir = Path(bundle)
        self.cache_dir = cache_dir
        self.chain = chain
        self.device = None
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
        # Timestep-only constants come from a subprocess, so the mod engines (schema 2)
        # never load here; schema 1 bundles compute theirs from the time engine.
        from ..vendor.groot_trt import load_step_constants
        if self.bundle.mod:
            load_step_constants(self.bundle, self.cache_dir)
        self.engines = Engines(self.bundle, self.cache_dir, skip=self.bundle.mod)
        self.fixture_parity = validate_fixture(self.bundle, self.engines)
        if self.fixture_parity["status"] != "PASS":
            raise ValueError(f"GR00T fixture parity failed: {self.fixture_parity}")
        if self.chain != "host":
            self._load_device()
            self.engines.release_buffers()

    def _load_device(self) -> None:
        """The device-resident chain must reproduce the host chain on the fixture inputs."""
        from ..vendor.groot_trt import _cmp, infer
        from ..vendor.trt_device import Groot16Device

        if self.n17:
            return self._load_device17()
        self.device = Groot16Device(self.bundle, self.engines, graph=self.chain == "graph")
        r = np.load(self.bundle.root / self.bundle.b["fixture"]["file"])
        args = (r["pixel_values"].astype(np.float32), r["state"], r["noise"])
        host = infer(self.bundle, self.engines.run, *args)
        dev = self.device.infer(*args)
        rep = _cmp(dev, host)
        rep["identical"] = bool(np.array_equal(dev, host))
        ok = _cmp(dev, r["action_pred"])
        self.fixture_parity["device_chain"] = {"vs_host_chain": rep, "chunk": ok}
        if ok["cosine"] < self.fixture_parity["threshold"] or \
                ok["max_pct_range"] > self.fixture_parity["max_pct_range_threshold"]:
            raise ValueError(f"device chain fails the fixture: {ok}")

    def _load_device17(self) -> None:
        """Fixture check through the device history: the fixture's earlier frame is fed
        first, then its current frame one lag later, so the history slot must pick it."""
        from ..vendor.groot17_trt import _unpatchify, encode_frames, infer
        from ..vendor.groot_trt import _cmp
        from ..vendor.trt_device import Groot17Device

        b, lag = self.bundle, self.history.lag_s
        self.device = Groot17Device(b, self.engines, lag, graph=self.chain == "graph")
        r = np.load(b.root / b.b["fixture"]["file"])
        thw = r["image_grid_thw"]
        pix = _unpatchify(r["pixel_values"].astype(np.float32), thw.shape[0],
                          int(thw[0, 1]), int(thw[0, 2]))
        v = b.b["views"]
        frames = [pix[i * v:(i + 1) * v] for i in range(b.b["frames"])]
        host = infer(b, self.engines.run, [encode_frames(b, self.engines.run, f)
                                           for f in frames], r["state"], r["noise"])
        self.device.infer(frames[0], r["state"], r["noise"], 0.0)
        dev, age = self.device.infer(frames[1], r["state"], r["noise"], lag)
        rep = _cmp(dev, host)
        rep["identical"] = bool(np.array_equal(dev, host))
        ok = _cmp(dev, r["action_pred"])
        self.fixture_parity["device_chain"] = {"vs_host_chain": rep, "chunk": ok,
                                               "history_age_s": age}
        # Start the benchmark with an empty history, like the host chain.
        from ..vendor.groot17_trt import FrameHistory
        self.device.history = FrameHistory(lag)
        if ok["cosine"] < self.fixture_parity["threshold"] or \
                ok["max_pct_range"] > self.fixture_parity["max_pct_range_threshold"]:
            raise ValueError(f"device chain fails the fixture: {ok}")

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
            "chain": self.chain,
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
        pv = np.stack(map_views(self.bundle.preprocess, obs.images))
        state = self._state(obs)
        pre = (time.perf_counter() - t0) * 1000
        t = {}
        if self.device is not None:
            actions = self.device.infer(pv, state, obs.noise, timings=t)
        else:
            actions = infer(self.bundle, self.engines.run, pv, state, obs.noise, timings=t)
        timings = {"total": pre + t["total"], "preprocess": pre,
                   "vision": t["vision"], "backbone": t["backbone"], "denoise": t["denoise"]}
        return InferResult(np.asarray(actions[0]), timings)

    def _infer17(self, obs: Observation) -> InferResult:
        from ..vendor.groot17_trt import encode_frames, infer

        t0 = time.perf_counter()
        pv = np.stack(map_views(self.bundle.preprocess, obs.images))
        state = self._state(obs)
        t1 = time.perf_counter()
        if self.device is not None:
            t = {}
            actions, age = self.device.infer(pv, state, obs.noise, t0, timings=t)
            self.history_ages_ms.append(age * 1000)
            pre = (t1 - t0) * 1000
            # Stage times are GPU events; the total is the host wall.
            timings = {"total": pre + t["total"], "preprocess": pre, "vision": t["vision"],
                       "backbone": t["backbone"], "denoise": t["denoise"]}
            return InferResult(np.asarray(actions[0]), timings)
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
