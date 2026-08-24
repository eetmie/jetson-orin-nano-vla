"""Backend: the split 9-graph ONNX export on ONNX Runtime + TensorRT EP (FP16).

This is the incumbent — the path already validated on this board and already driving
the excavator (`kaivuriprokkis/lerobot_vla/smolvla_split.py`, vendored under
`bench/vendor/`). It exists because the *monolithic* SmolVLA ONNX cannot TRT-build in
8 GB: TensorRT imports all 450M weights as FP32 working copies at once, a ~6 GB floor
that is independent of node count. Splitting the model means each engine carries only
its own weight slice.

What that costs, and why this backend is instrumented per-graph
---------------------------------------------------------------
Three graphs run on TensorRT (vision, expert-prefill, expert-decode). The other six —
the text encoder and five projectors — run on the **CPU** execution provider, and the
flow-matching denoise loop itself is numpy in Python: for every one of the 10 steps it
runs action_in, time_in, time_out and action_out on the CPU and does the Euler update
in numpy. On a board where the whole point is to leave CPU headroom for the robot
control stack, that is exactly the thing to measure rather than assume. So each ORT
session is wrapped in a timer and the per-graph split is reported alongside the total.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from .base import Backend, InferResult, bundle_camera_key, load_export_info, load_stats


class _TimedSession:
    """Delegating wrapper that accumulates wall time per ORT session.

    Wrapping rather than editing the vendored module keeps the measured code
    identical to what ships on the robot.
    """

    def __init__(self, sess, label: str, sink: dict):
        self._s, self._label, self._sink = sess, label, sink

    def run(self, output_names, input_feed, run_options=None):
        t0 = time.perf_counter()
        out = self._s.run(output_names, input_feed, run_options)
        self._sink[self._label] = self._sink.get(self._label, 0.0) + \
            (time.perf_counter() - t0) * 1000.0
        self._sink[self._label + "_n"] = self._sink.get(self._label + "_n", 0) + 1
        return out

    def __getattr__(self, item):
        return getattr(self._s, item)


class OrtSplitBackend(Backend):
    name = "ort-split-trt"
    noise_injected = True

    _SESSIONS = ("vision", "prefill", "decode", "text", "state_proj",
                 "action_in", "action_out", "time_in", "time_out")
    _GPU_GRAPHS = ("vision", "prefill", "decode")

    def __init__(self, bundle: Path, cache_dir: str, precision: str = "fp16",
                 num_steps: int | None = None, action_dim: int = 4,
                 drop_cuda_ep: bool = False, seed: int = 0):
        self.bundle = Path(bundle)
        self.cache_dir = cache_dir
        self.precision = precision
        self.action_dim = action_dim
        self.seed = seed
        self._info = load_export_info(self.bundle)
        self.num_steps = num_steps or int(self._info.get("num_steps", 10))
        if drop_cuda_ep:
            # The CUDA EP holds a 3 GiB arena. Dropping it frees memory for the TRT
            # build on a tight board, at the cost of any TRT-rejected op landing on
            # the CPU instead of CUDA. `TRT_DROP_CUDA_EP=1` is the documented recipe.
            os.environ["TRT_DROP_CUDA_EP"] = "1"
        self.policy = None
        self._sink: dict = {}

    def load(self) -> None:
        from ..vendor.smolvla_split import SmolVLASplitPolicy

        tok = self.bundle / "tokenizer"
        self.policy = SmolVLASplitPolicy(
            split_dir=self.bundle,
            tokenizer_dir=tok if tok.exists() else self.bundle,
            cache_dir=self.cache_dir,
            precision=self.precision,
            num_steps=self.num_steps,
            action_dim=self.action_dim,
            norm=load_stats(self.bundle),
            seed=self.seed,
        )
        self._providers = {n: getattr(self.policy, n).get_providers()[0]
                           for n in self._SESSIONS}
        for n in self._SESSIONS:
            setattr(self.policy, n, _TimedSession(getattr(self.policy, n), n, self._sink))

    def meta(self) -> dict:
        import onnxruntime as ort
        p = self.policy
        return {
            "backend": self.name,
            "bundle": str(self.bundle),
            "precision": self.precision,
            "num_steps": self.num_steps,
            "chunk_size": getattr(p, "chunk_size", None),
            "prefix_len": getattr(p, "prefix_len", None),
            "n_cam_slots": getattr(p, "n_cam_slots", None),
            "action_dim": self.action_dim,
            "providers_per_graph": self._providers,
            "graphs_on_gpu": [n for n in self._GPU_GRAPHS
                              if self._providers.get(n) == "TensorrtExecutionProvider"],
            "onnxruntime": ort.__version__,
            "engine_cache": self.cache_dir,
            "cuda_ep_dropped": bool(os.environ.get("TRT_DROP_CUDA_EP")),
            "camera_key": bundle_camera_key(self.bundle),
            "export_info": self._info,
        }

    def infer(self, obs: Observation) -> InferResult:
        self._sink.clear()
        t0 = time.perf_counter()
        chunk = self.policy.sample_actions(obs.image, obs.task, obs.state,
                                           noise=obs.noise)
        total = (time.perf_counter() - t0) * 1000.0

        t = {"total": total}
        gpu = cpu = 0.0
        for n in self._SESSIONS:
            ms = self._sink.get(n, 0.0)
            t[f"graph.{n}"] = round(ms, 3)
            t[f"graph.{n}.calls"] = self._sink.get(n + "_n", 0)
            if self._providers.get(n) in ("TensorrtExecutionProvider",
                                          "CUDAExecutionProvider"):
                gpu += ms
            else:
                cpu += ms
        t["graphs_gpu"] = round(gpu, 3)
        t["graphs_cpu"] = round(cpu, 3)
        # Whatever is left is numpy: masks, the Euler update, resize, tokenization.
        t["python_numpy"] = round(total - gpu - cpu, 3)
        return InferResult(np.asarray(chunk), t)
