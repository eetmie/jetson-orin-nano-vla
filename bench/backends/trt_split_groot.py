"""GR00T N1.6 split bundle on the TensorRT runtime (no ONNX Runtime)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from .base import Backend, InferResult


class TrtSplitGrootBackend(Backend):
    name = "trt-split-groot"
    noise_injected = True

    def __init__(self, bundle: Path, cache_dir: str) -> None:
        self.bundle_dir = Path(bundle)
        self.cache_dir = cache_dir
        self.bundle = None
        self.engines = None

    def load(self) -> None:
        from ..vendor.groot_trt import Bundle, Engines, prebuild_engines, validate_fixture

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
        return {
            "backend": self.name,
            "family": "groot",
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

    def infer(self, obs: Observation) -> InferResult:
        from ..vendor.groot_trt import infer

        b = self.bundle.b
        if len(obs.images) != b["views"]:
            raise ValueError(f"this GR00T bundle was exported for {b['views']} view(s), "
                             f"got {len(obs.images)}")
        if obs.task != b["task"]:
            raise ValueError(f"the bundle bakes the prompt {b['task']!r}; got {obs.task!r}. "
                             "Re-export with --task to change it.")
        t0 = time.perf_counter()
        pv = np.stack([self.bundle.preprocess(im) for im in obs.images])
        # The stock processor min-max normalizes state and clips it to [-1, 1]; the
        # synthetic stream's raw N(0, 20^2) state is put in that same range.
        state = np.zeros((1, 1, b["state_dim"]), np.float32)
        state[0, 0, :obs.state.shape[0]] = np.clip(obs.state, -1.0, 1.0)
        pre = (time.perf_counter() - t0) * 1000
        t = {}
        actions = infer(self.bundle, self.engines.run, pv, state, obs.noise, timings=t)
        timings = {"total": pre + t["total"], "preprocess": pre,
                   "vision": t["vision"], "backbone": t["backbone"], "denoise": t["denoise"]}
        return InferResult(np.asarray(actions[0]), timings)
