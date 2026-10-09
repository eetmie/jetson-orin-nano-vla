"""SmolVLA split bundles on the TensorRT runtime (no ONNX Runtime).

Same graphs and host loop as `ort-split`, with every graph an engine and the token
embedding memory-mapped. The bundle must come from export/export.sh, which ships the
mixed-FP16 heavy graphs, `embed_tokens.npy` and the stock-PyTorch fixture the engines are
checked against at load; the run fails closed if they disagree.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from .base import Backend, InferResult, load_stats


class TrtSplitSmolVLABackend(Backend):
    name = "trt-split-smolvla"
    noise_injected = True

    def __init__(self, bundle: Path, cache_dir: str, action_dim: int = 32) -> None:
        self.bundle_dir = Path(bundle)
        self.cache_dir = cache_dir
        self.action_dim = action_dim
        self.bundle = None
        self.engines = None

    def load(self) -> None:
        from ..vendor.groot_trt import Engines, prebuild_engines
        from ..vendor.smolvla_trt import Bundle, Policy, validate_fixture

        self.bundle = Bundle(self.bundle_dir)
        if not (self.bundle.info.get("fixture") or {}).get("file"):
            raise ValueError(f"{self.bundle_dir} carries no reference fixture: export it "
                             "with export/export.sh (Hugging Face bundles run on ort-split)")
        self.norm = load_stats(self.bundle_dir)
        self.built = prebuild_engines(self.bundle, self.cache_dir)
        self.engines = Engines(self.bundle, self.cache_dir)
        self.policy = Policy(self.bundle, self.engines.run)
        self.fixture_parity = validate_fixture(self.bundle, self.policy)
        if self.fixture_parity["status"] != "PASS":
            raise ValueError(f"SmolVLA fixture parity failed: {self.fixture_parity}")

    def artifact_paths(self) -> dict[str, Path]:
        return {"bundle": self.bundle_dir}

    def meta(self) -> dict:
        import tensorrt as trt

        b = self.bundle
        return {
            "backend": self.name,
            "family": "smolvla",
            "bundle": str(self.bundle_dir),
            "precision": "mixed fp16 (strongly typed)",
            "num_steps": b.num_steps,
            "chunk_size": b.chunk_size,
            "prefix_len": b.prefix_len,
            "n_cam_slots": b.n_cam_slots,
            "action_dim": self.action_dim,
            "resize": [512, 512],
            "n_graphs": len(b.names),
            "configured_provider_priority_per_graph": {
                n: "TensorrtExecutionProvider" for n in b.names},
            "runtime": "tensorrt python, no onnxruntime",
            "tensorrt": trt.__version__,
            "shared_scratch_mb": round(self.engines.scratch_bytes / 2**20, 1),
            "token_embedding": "fp32 .npy, memory-mapped on the CPU",
            "engine_cache": str(Path(self.cache_dir).expanduser()),
            "engines_built_this_load_s": self.built,
            "fixture_parity": self.fixture_parity,
            "export_info": {k: v for k, v in b.info.items() if k != "fixture"},
        }

    def infer(self, obs: Observation) -> InferResult:
        from ..vendor.smolvla_trt import pad_state, preprocess

        b, p = self.bundle, self.policy
        if len(obs.images) > b.n_cam_slots:
            raise ValueError(f"{len(obs.images)} cameras given but the export has "
                             f"{b.n_cam_slots} camera slot(s)")
        t0 = time.perf_counter()
        pix = [preprocess(im) for im in obs.images]
        lang = b.language(obs.task)
        state = pad_state(self.norm.normalize_state(
            np.asarray(obs.state, np.float32).reshape(-1)))
        t1 = time.perf_counter()
        embs = [p.vision(x) for x in pix]
        t2 = time.perf_counter()
        t = {}
        x_t = p.sample(embs, lang, state, obs.noise, timings=t)
        chunk = self.norm.unnormalize_action(x_t[0, :, :self.action_dim])
        pre, vis = (t1 - t0) * 1000, (t2 - t1) * 1000
        timings = {"total": pre + vis + t["prefill"] + t["denoise"], "preprocess": pre,
                   "vision": vis, "prefill": t["prefill"], "denoise": t["denoise"]}
        return InferResult(np.asarray(chunk), timings)
