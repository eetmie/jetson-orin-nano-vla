"""X-VLA split bundles on the TensorRT runtime (no ONNX Runtime).

Same graphs and host loop as `ort-split`, with each weight held once. The bundle must
carry the exporter's stock-PyTorch fixture: the engines are checked against it at load
and the run fails closed if they disagree.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from ..vendor.imaging import map_views
from .base import Backend, InferResult


class TrtSplitXVLABackend(Backend):
    name = "trt-split-xvla"
    noise_injected = True

    def __init__(self, bundle: Path, cache_dir: str, chain: str = "host") -> None:
        self.bundle_dir = Path(bundle)
        self.cache_dir = cache_dir
        self.chain = chain
        self.device = None
        self.bundle = None
        self.engines = None

    def load(self) -> None:
        from ..vendor.groot_trt import Engines, prebuild_engines
        from ..vendor.xvla_trt import Bundle, validate_fixture

        self.bundle = Bundle(self.bundle_dir)
        if not (self.bundle.b.get("fixture") or {}).get("file"):
            raise ValueError(f"{self.bundle_dir} carries no reference fixture: export it "
                             "with export/export.sh (Hugging Face bundles run on ort-split)")
        self.built = prebuild_engines(self.bundle, self.cache_dir)
        self.engines = Engines(self.bundle, self.cache_dir)
        self.fixture_parity = validate_fixture(self.bundle, self.engines)
        if self.fixture_parity["status"] != "PASS":
            raise ValueError(f"X-VLA fixture parity failed: {self.fixture_parity}")
        if self.chain != "host":
            self._load_device()
            self.engines.release_buffers()

    def _load_device(self) -> None:
        """The device-resident chain must reproduce the host chain on the fixture inputs."""
        from ..vendor.groot_trt import _cmp
        from ..vendor.trt_device import XVLADevice
        from ..vendor.trt_ops import Ops
        from ..vendor.xvla_trt import infer

        self.device = XVLADevice(self.bundle, self.engines, Ops(self.engines, self.cache_dir),
                                 graph=self.chain == "graph")
        r = np.load(self.bundle.root / self.bundle.b["fixture"]["file"])
        args = (r["pixel_values"].astype(np.float32), r["input_ids"].astype(np.int64),
                r["proprio"].astype(np.float32), r["x1"])
        host = infer(self.bundle, self.engines.run, *args)
        dev = self.device.infer(*args)
        rep = _cmp(dev, host)
        rep["identical"] = bool(np.array_equal(dev, host))
        ok = _cmp(dev, r["action_pred"])
        self.fixture_parity["device_chain"] = {"vs_host_chain": rep, "chunk": ok}
        if ok["cosine"] < self.fixture_parity["threshold"] or \
                ok["max_pct_range"] > self.fixture_parity["max_pct_range_threshold"]:
            raise ValueError(f"device chain fails the fixture: {ok}")

    def artifact_paths(self) -> dict[str, Path]:
        return {"bundle": self.bundle_dir}

    def meta(self) -> dict:
        import tensorrt as trt

        bd, b = self.bundle, self.bundle.b
        return {
            "backend": self.name,
            "family": "xvla",
            "bundle": str(self.bundle_dir),
            "precision": "mixed fp16 (strongly typed)",
            "num_steps": bd.steps,
            "chunk_size": bd.chunk_size,
            "action_dim": bd.action_dim,
            "state_dim": bd.state_dim,
            "num_views": bd.num_views,
            "valid_views": bd.valid_views,
            "processed_views": bd.valid_views,
            "resize": [224, 224],
            "tokens_per_view": bd.tokens_per_view,
            "lang_len": bd.lang_len,
            "action_mode": b.get("action_mode"),
            "domain_id": b.get("domain_id"),
            "n_graphs": len(bd.names),
            "configured_provider_priority_per_graph": {
                n: "TensorrtExecutionProvider" for n in bd.names},
            "runtime": "tensorrt python, no onnxruntime",
            "tensorrt": trt.__version__,
            "shared_scratch_mb": round(self.engines.scratch_bytes / 2**20, 1),
            "engine_cache": str(Path(self.cache_dir).expanduser()),
            "engines_built_this_load_s": self.built,
            "fixture_parity": self.fixture_parity,
            "chain": self.chain,
            "kv_cache": False,
            "kv_cache_note": "impossible — bidirectional policy transformer, "
                             "conditioning attends to action tokens and changes per step",
        }

    def infer(self, obs: Observation) -> InferResult:
        from ..vendor.xvla_trt import infer

        bd = self.bundle
        if len(obs.images) != bd.valid_views:
            raise ValueError(f"observation has {len(obs.images)} view(s), bundle requires "
                             f"exactly {bd.valid_views}")
        t0 = time.perf_counter()
        pv = np.stack(map_views(bd.preprocess, obs.images))
        ids = bd.input_ids(obs.task)
        proprio = np.zeros((1, bd.state_dim), np.float32)
        flat = np.asarray(obs.state, np.float32).ravel()[:bd.state_dim]
        proprio[0, :len(flat)] = flat
        pre = (time.perf_counter() - t0) * 1000
        t = {}
        # X-VLA injects x1, the single fixed draw the loop interpolates against.
        if self.device is not None:
            chunk = self.device.infer(pv, ids, proprio, obs.noise, timings=t)
        else:
            chunk = infer(bd, self.engines.run, pv, ids, proprio, obs.noise, timings=t)
        # The device chain's stage times are GPU events; its total is the host wall.
        total = t.get("total", t["vision"] + t["text"] + t["cond"] + t["denoise"])
        timings = {"total": pre + total,
                   "preprocess": pre, "vision": t["vision"], "text": t["text"],
                   "cond": t["cond"], "denoise": t["denoise"]}
        return InferResult(np.asarray(chunk[0]), timings)
