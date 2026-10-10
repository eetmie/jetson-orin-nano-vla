"""pi0.5 split bundles on the TensorRT runtime (prototype).

Refitted layer templates, one device-resident chain, CUDA-graph replay by default. The
bundle carries full-FP16 PyTorch reference trajectories; load runs them and refuses to
benchmark on a miss.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from ..vendor.imaging import map_views
from .base import Backend, InferResult


class TrtSplitPi05Backend(Backend):
    name = "trt-split-pi05"
    noise_injected = True

    def __init__(self, bundle: Path, cache_dir: str, chain: str = "graph") -> None:
        self.bundle_dir = Path(bundle)
        self.cache_dir = cache_dir
        # The prototype chain is device-resident by construction; there is no numpy one.
        self.chain = "device" if chain == "host" else chain
        self.bundle = None

    def load(self) -> None:
        from ..vendor.pi05_trt import (Bundle, Pi05Device, Pi05Engines, prebuild_templates,
                                       validate_fixtures)
        from ..vendor.trt_ops import Ops

        self.bundle = Bundle(self.bundle_dir)
        self.built = prebuild_templates(self.bundle, self.cache_dir)
        t0 = time.perf_counter()
        self.engines = Pi05Engines(self.bundle, self.cache_dir)
        self.refit_s = round(time.perf_counter() - t0, 1)
        self.device = Pi05Device(self.bundle, self.engines, Ops(self.engines, self.cache_dir),
                                 graph=self.chain == "graph")
        self.fixture_parity = validate_fixtures(self.bundle, self.device)
        if self.fixture_parity["status"] != "PASS":
            raise ValueError(f"pi0.5 fixture parity failed: {self.fixture_parity['worst']}")

    def artifact_paths(self) -> dict[str, Path]:
        return {"bundle": self.bundle_dir}

    def meta(self) -> dict:
        import tensorrt as trt

        b = self.bundle
        fp = dict(self.fixture_parity)
        fp.pop("reports", None)
        return {
            "backend": self.name,
            "family": "pi05",
            "bundle": str(self.bundle_dir),
            "precision": b.b.get("dtype"),
            "layout": "compact prefix" if b.compact else "padded prefix",
            "prefix_tokens": b.prefix,
            "num_steps": b.steps,
            "chunk_size": b.horizon,
            "action_dim": b.width,
            "robot_action_dim": int(b.b["robot_action_dim"]),
            "image_slots": b.slots,
            "components": len(b.names),
            "templates": b.kinds,
            "runtime": "tensorrt python, refitted stripped templates, no onnxruntime",
            "tensorrt": trt.__version__,
            "chain": self.chain,
            "shared_scratch_mb": round(self.engines.scratch_bytes / 2**20, 1),
            "trt_allocator_live_mb": round(sum(self.engines.allocator.live.values()) / 2**20, 1),
            "engine_cache": str(Path(self.cache_dir).expanduser()),
            "templates_built_this_load_s": self.built,
            "refit_load_s": self.refit_s,
            "fixture_parity": fp,
        }

    def infer(self, obs: Observation) -> InferResult:
        b = self.bundle
        if len(obs.images) != int(b.b["valid_cameras"]):
            raise ValueError(f"pi0.5 LIBERO takes {b.b['valid_cameras']} cameras (base and "
                             f"wrist), got {len(obs.images)}")
        t0 = time.perf_counter()
        imgs = map_views(b.preprocess, obs.images)
        ids, mask = b.tokens(obs.task)
        pre = (time.perf_counter() - t0) * 1000
        t = {}
        act = self.device.infer(imgs, ids, mask, obs.noise.astype(np.float32), timings=t)
        robot = b.unnormalize(act[0])
        # Stage times are GPU events; the total is the host wall.
        timings = {"total": pre + t["total"], "preprocess": pre, "vision": t["vision"],
                   "language": t["language"], "denoise": t["denoise"]}
        return InferResult(np.asarray(robot), timings)
