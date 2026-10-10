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


def _accumulate() -> str:
    from ..vendor import groot_trt
    return groot_trt.ACCUMULATE


class TrtSplitSmolVLABackend(Backend):
    name = "trt-split-smolvla"
    noise_injected = True

    def __init__(self, bundle: Path, cache_dir: str, action_dim: int = 32,
                 chain: str = "host") -> None:
        self.bundle_dir = Path(bundle)
        self.cache_dir = cache_dir
        self.chain = chain
        self.device = None
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
        if self.chain != "host":
            self._load_device()
            self.engines.release_buffers()

    def _load_device(self) -> None:
        """The device-resident chain must reproduce the host chain on the fixture inputs."""
        from ..vendor.groot_trt import _cmp
        from ..vendor.trt_device import SmolVLADevice
        from ..vendor.trt_ops import Ops

        b, p = self.bundle, self.policy
        self.device = SmolVLADevice(b, self.engines, Ops(self.engines, self.cache_dir),
                                    graph=self.chain == "graph")
        r = np.load(b.root / b.info["fixture"]["file"])
        pix = r["pixel_values"].astype(np.float32)
        lang = (b.embed_ids(r["lang_tokens"]), r["lang_masks"].astype(bool))
        state = r["model_state"].astype(np.float32)
        host = p.sample([p.vision(pix[i:i + 1]) for i in range(len(pix))], lang, state,
                        r["noise"])
        dev = self.device.infer([pix[i:i + 1] for i in range(len(pix))], lang, state,
                                r["noise"], key=("fixture",))
        rep = _cmp(dev, host)
        rep["identical"] = bool(np.array_equal(dev, host))
        # The uint8 path must give the vision engine the host conversion's exact floats.
        from ..vendor.smolvla_split import resize_pad_canvas, siglip_normalize
        img = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        canvas = resize_pad_canvas(img)
        d = self.device.d
        d.upload(self.device.canvas[0], canvas)
        d.enqueue(self.device.to_pix, {"x": self.device.canvas[0], "out": self.device.pix[0]})
        d.download(self.device.pix[0])
        d.sync()
        gpu = self.device.pix[0].host().copy()
        if not np.array_equal(gpu, siglip_normalize(canvas)):
            raise ValueError("GPU uint8 -> SigLIP conversion differs from the host one")
        self.fixture_parity["gpu_image_conversion"] = "identical to host"
        ok = _cmp(dev, r["action_pred"])
        self.fixture_parity["device_chain"] = {"vs_host_chain": rep, "chunk": ok}
        if ok["cosine"] < self.fixture_parity["threshold"] or \
                ok["max_pct_range"] > self.fixture_parity["max_pct_range_threshold"]:
            raise ValueError(f"device chain fails the fixture: {ok}")

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
            "accumulate": _accumulate(),
            "tensorrt": trt.__version__,
            "shared_scratch_mb": round(self.engines.scratch_bytes / 2**20, 1),
            "token_embedding": "fp32 .npy, memory-mapped on the CPU",
            "engine_cache": str(Path(self.cache_dir).expanduser()),
            "engines_built_this_load_s": self.built,
            "fixture_parity": self.fixture_parity,
            "chain": self.chain,
            # Not model_id: the exporter records the export machine's checkpoint path.
            "export_info": {k: v for k, v in b.info.items()
                            if k not in ("fixture", "model_id")},
        }

    def infer(self, obs: Observation) -> InferResult:
        from ..vendor.smolvla_trt import pad_state, preprocess

        b, p = self.bundle, self.policy
        if len(obs.images) > b.n_cam_slots:
            raise ValueError(f"{len(obs.images)} cameras given but the export has "
                             f"{b.n_cam_slots} camera slot(s)")
        t0 = time.perf_counter()
        if self.device is not None:
            from ..vendor.smolvla_split import resize_pad_canvas
            # Resize + pad on the CPU straight into the pinned upload buffer; the float
            # conversion runs on the GPU.
            pix = [resize_pad_canvas(im, out=self.device.canvas_buffer(i))
                   for i, im in enumerate(obs.images)]
        else:
            pix = [preprocess(im) for im in obs.images]
        lang = b.language(obs.task)
        state = pad_state(self.norm.normalize_state(
            np.asarray(obs.state, np.float32).reshape(-1)))
        t1 = time.perf_counter()
        t = {}
        if self.device is not None:
            # Stage times are GPU events; the total is the host wall.
            x_t = self.device.infer(pix, lang, state, obs.noise, key=obs.task, timings=t)
            pre, vis, total = (t1 - t0) * 1000, t["vision"], t["total"]
        else:
            embs = [p.vision(x) for x in pix]
            t2 = time.perf_counter()
            x_t = p.sample(embs, lang, state, obs.noise, timings=t)
            pre, vis = (t1 - t0) * 1000, (t2 - t1) * 1000
            total = vis + t["prefill"] + t["denoise"]
        chunk = self.norm.unnormalize_action(x_t[0, :, :self.action_dim])
        timings = {"total": pre + total, "preprocess": pre,
                   "vision": vis, "prefill": t["prefill"], "denoise": t["denoise"]}
        return InferResult(np.asarray(chunk), timings)
