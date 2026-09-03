"""Backend for the nondeployable EVO1 eleven-graph bootstrap bundle."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..evo1_parity import validate_fixture
from ..obs import Observation
from .base import Backend, InferResult, tree_sha256


class OrtSplitEvo1Backend(Backend):
    name = "ort-split-evo1"
    noise_injected = True

    def __init__(
        self,
        bundle: Path,
        cache_dir: str,
        precision: str = "fp16",
        num_steps: int | None = None,
    ) -> None:
        self.bundle = Path(bundle)
        self.cache_dir = cache_dir
        self.precision = precision
        self.num_steps = num_steps
        self.policy = None

    def load(self) -> None:
        from ..vendor.evo1_split_ort import Evo1SplitPolicy, prebuild_engines

        prebuild_engines(self.bundle, self.cache_dir, self.precision)
        self.policy = Evo1SplitPolicy(
            self.bundle,
            self.cache_dir,
            self.precision,
            num_steps=self.num_steps,
            allow_bootstrap=True,
        )
        self.fixture_parity = validate_fixture(self.policy, self.bundle)
        if self.fixture_parity["status"] != "PASS":
            raise ValueError("EVO1 native fixture parity failed")
        self._providers = {
            name: session.get_providers()[0]
            for name, session in self.policy.sessions.items()
        }

    def artifact_paths(self) -> dict[str, Path]:
        return {"bundle": self.bundle}

    def meta(self) -> dict:
        import onnxruntime as ort

        policy = self.policy
        bundle = policy.bundle
        tokenizer_path = self.bundle / bundle["tokenizer"]["path"]
        return {
            "backend": self.name,
            "family": "evo1",
            "bundle": str(self.bundle),
            "precision": self.precision,
            "num_steps": policy.steps,
            "chunk_size": int(bundle["chunk_size"]),
            "state_dim": int(bundle["max_state_dim"]),
            "action_dim": int(bundle["max_action_dim"]),
            "views": int(bundle["valid_views"]),
            "resize": [int(bundle["image_size"])] * 2,
            "sequence_length": int(bundle["seq_len"]),
            "tokenizer": str(tokenizer_path),
            "tokenizer_sha256": tree_sha256(tokenizer_path),
            "n_graphs": len(policy.sessions),
            "n_graphs_on_trt": sum(
                provider == "TensorrtExecutionProvider"
                for provider in self._providers.values()
            ),
            "configured_provider_priority_per_graph": self._providers,
            "onnxruntime": ort.__version__,
            "engine_cache": self.cache_dir,
            "token_embedding_device": "cpu",
            "device_resident_action_cache_and_hidden": True,
            "host_euler_update": True,
            "cuda_fallback": False,
            "deployable": False,
            "random_action_head": True,
            "warning": bundle["warning"],
            "base": bundle["base"],
            "provenance": bundle["provenance"],
            "fixture_parity": self.fixture_parity,
        }

    def infer(self, obs: Observation) -> InferResult:
        if len(obs.images) != 1:
            raise ValueError(
                f"EVO1 bootstrap bundle requires one image, got {len(obs.images)}"
            )
        started = time.perf_counter()
        chunk = self.policy.sample_actions(
            obs.image,
            obs.task,
            obs.state,
            noise=obs.noise,
        )
        total = (time.perf_counter() - started) * 1000.0
        timings = {"total": total}
        for name, seconds in self.policy.last_timings.items():
            timings[f"runtime.{name}"] = round(float(seconds) * 1000.0, 3)
        timings["python_numpy"] = round(
            total - self.policy.last_timings.get("total", 0.0) * 1000.0, 3
        )
        return InferResult(np.asarray(chunk), timings)
