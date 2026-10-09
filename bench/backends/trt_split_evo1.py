"""EVO1 split bundles on the TensorRT runtime (no ONNX Runtime).

Same graphs and host loop as `ort-split`, with every graph an engine and the token
embedding memory-mapped. The bundle must come from export/export.sh, which ships
`embed_tokens.npy` and the native-LeRobot fixture the engines are checked against at
load; the run fails closed if they disagree.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from .base import Backend, InferResult


class TrtSplitEvo1Backend(Backend):
    name = "trt-split-evo1"
    noise_injected = True

    def __init__(self, bundle: Path, cache_dir: str) -> None:
        self.bundle_dir = Path(bundle)
        self.cache_dir = cache_dir
        self.bundle = None
        self.engines = None

    def load(self) -> None:
        from ..vendor.evo1_trt import Bundle, validate_fixture
        from ..vendor.groot_trt import Engines, prebuild_engines

        self.bundle = Bundle(self.bundle_dir)
        if not (self.bundle.b.get("fixture") or {}).get("file"):
            raise ValueError(f"{self.bundle_dir} carries no reference fixture: export it "
                             "with export/export.sh (Hugging Face bundles run on ort-split)")
        self.built = prebuild_engines(self.bundle, self.cache_dir)
        self.engines = Engines(self.bundle, self.cache_dir)
        self.fixture_parity = validate_fixture(self.bundle, self.engines)
        if self.fixture_parity["status"] != "PASS":
            raise ValueError(f"EVO1 fixture parity failed: {self.fixture_parity}")

    def artifact_paths(self) -> dict[str, Path]:
        return {"bundle": self.bundle_dir}

    def meta(self) -> dict:
        import tensorrt as trt

        bd, b = self.bundle, self.bundle.b
        return {
            "backend": self.name,
            "family": "evo1",
            "bundle": str(self.bundle_dir),
            "precision": "mixed fp16 (strongly typed)",
            "num_steps": bd.steps,
            "chunk_size": int(b["chunk_size"]),
            "state_dim": int(b["max_state_dim"]),
            "action_dim": int(b["max_action_dim"]),
            "views": bd.views,
            "resize": [int(b["image_size"])] * 2,
            "sequence_length": int(b["seq_len"]),
            "n_graphs": len(bd.names),
            "configured_provider_priority_per_graph": {
                n: "TensorrtExecutionProvider" for n in bd.names},
            "runtime": "tensorrt python, no onnxruntime",
            "tensorrt": trt.__version__,
            "shared_scratch_mb": round(self.engines.scratch_bytes / 2**20, 1),
            "token_embedding": "fp32 .npy, memory-mapped on the CPU",
            "engine_cache": str(Path(self.cache_dir).expanduser()),
            "engines_built_this_load_s": self.built,
            "deployable": bool(b.get("deployable")),
            "random_action_head": bool(b.get("random_action_head")),
            "warning": b.get("warning"),
            "base": b.get("base"),
            "provenance": b.get("provenance"),
            "fixture_parity": self.fixture_parity,
        }

    def infer(self, obs: Observation) -> InferResult:
        from ..vendor.evo1_trt import infer

        bd, b = self.bundle, self.bundle.b
        if len(obs.images) != bd.views:
            raise ValueError(f"this EVO1 bundle declares {bd.views} view(s), got "
                             f"{len(obs.images)}. A padded view still spends its image "
                             "tokens, so the counts must match.")
        t0 = time.perf_counter()
        pv = bd.preprocess(obs.images)
        ids, cmask = bd.prompt(obs.task)
        width = int(b["max_state_dim"])
        state = np.zeros((1, width), np.float32)
        flat = np.asarray(obs.state, np.float32).reshape(-1)
        if flat.size > width:
            raise ValueError(f"state has {flat.size} values, bundle supports {width}")
        state[0, :flat.size] = flat
        pre = (time.perf_counter() - t0) * 1000
        t = {}
        out = infer(bd, self.engines.run, pv, ids, cmask, state, obs.noise, timings=t)
        timings = {"total": pre + sum(t.values()), "preprocess": pre, **t}
        return InferResult(np.asarray(out["action"][0]), timings)
