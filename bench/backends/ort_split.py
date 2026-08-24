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

#: Graphs the GPU stack refused, recorded rather than raised — see _move_projectors_to_gpu.
LOG_MOVE_FAILURES: list[str] = []


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


#: Attribute name -> ONNX file, for the graphs the stock runtime puts on the CPU EP.
#: Needed to re-create them on the GPU stack under `--projectors gpu`.
_CPU_GRAPH_FILES = {
    "text": "smolvlm_text.onnx",
    "state_proj": "state_projector.onnx",
    "action_in": "action_in_projector.onnx",
    "action_out": "action_out_projector.onnx",
    "time_in": "time_in_projector.onnx",
    "time_out": "time_out_projector.onnx",
}

#: Called once per denoising step, so ten times per inference at the default budget.
#: These are the four that make the CPU-side loop worth measuring.
_PER_STEP_PROJECTORS = ("action_in", "time_in", "time_out", "action_out")


class OrtSplitBackend(Backend):
    name = "ort-split-trt"
    noise_injected = True

    _SESSIONS = ("vision", "prefill", "decode", "text", "state_proj",
                 "action_in", "action_out", "time_in", "time_out")
    _GPU_GRAPHS = ("vision", "prefill", "decode")

    def __init__(self, bundle: Path, cache_dir: str, precision: str = "fp16",
                 num_steps: int | None = None, action_dim: int = 4,
                 drop_cuda_ep: bool = False, seed: int = 0,
                 projectors: str = "cpu", trt_opt_level: int | None = None,
                 trt_workspace_mb: int | None = None,
                 tokenizer: str | None = None):
        self.bundle = Path(bundle)
        self.cache_dir = cache_dir
        self.precision = precision
        self.action_dim = action_dim
        self.seed = seed
        self.projectors = projectors
        self.tokenizer = tokenizer
        self._info = load_export_info(self.bundle)
        self.num_steps = num_steps or int(self._info.get("num_steps", 10))
        self._moved: list[str] = []
        if drop_cuda_ep:
            # The CUDA EP holds a 3 GiB arena. Dropping it frees memory for the TRT
            # build on a tight board, at the cost of any TRT-rejected op landing on
            # the CPU instead of CUDA. `TRT_DROP_CUDA_EP=1` is the documented recipe.
            os.environ["TRT_DROP_CUDA_EP"] = "1"
        # The vendored builder reads both from the environment, and its own defaults
        # (level 2, 512 MB) are the ones proven to fit 8 GB. Raising them is a
        # one-time build cost for a possibly faster cached engine — worth an A/B, but
        # clear the engine cache between attempts or nothing rebuilds.
        if trt_opt_level is not None:
            os.environ["TRT_OPT_LEVEL"] = str(trt_opt_level)
        if trt_workspace_mb is not None:
            os.environ["TRT_WORKSPACE_MB"] = str(trt_workspace_mb)
        self.trt_opt_level = trt_opt_level
        self.trt_workspace_mb = trt_workspace_mb
        self.policy = None
        self._sink: dict = {}

    def load(self) -> None:
        from ..vendor.smolvla_split import SmolVLASplitPolicy

        tok = Path(self.tokenizer) if self.tokenizer else self.bundle / "tokenizer"
        self.policy = SmolVLASplitPolicy(
            split_dir=self.bundle,
            tokenizer_dir=tok if Path(tok).exists() else (self.tokenizer or self.bundle),
            cache_dir=self.cache_dir,
            precision=self.precision,
            num_steps=self.num_steps,
            action_dim=self.action_dim,
            norm=load_stats(self.bundle),
            seed=self.seed,
        )
        if self.projectors == "gpu":
            self._move_projectors_to_gpu()
        self._providers = {n: getattr(self.policy, n).get_providers()[0]
                           for n in self._SESSIONS}
        for n in self._SESSIONS:
            setattr(self.policy, n, _TimedSession(getattr(self.policy, n), n, self._sink))

    def _move_projectors_to_gpu(self) -> None:
        """Re-create the per-step projectors on the TensorRT/CUDA stack.

        The stock runtime puts the text encoder and all five projectors on the CPU EP
        because they are tiny. Tiny per call, but `action_in`, `time_in`, `time_out`
        and `action_out` each run once per denoising step — ten times per inference —
        and every one is a host round trip in the middle of a GPU loop. Whether that
        is 30-60 ms or a rounding error is a measurement, which is what this flag and
        the per-graph timings exist to settle.

        Done here, by rebuilding the sessions after construction, rather than by
        editing the vendored runtime: the vendored file must stay the code that runs
        on the robot. Four extra TRT builds are added; if TRT declines a graph the EP
        stack falls through to CUDA on its own, and `providers_per_graph` records
        wherever each one actually landed.
        """
        import onnxruntime as ort
        from ..vendor.smolvla_split import build_providers

        heavy = build_providers(self.cache_dir, precision=self.precision)
        for name in _PER_STEP_PROJECTORS:
            path = self.bundle / _CPU_GRAPH_FILES[name]
            if not path.exists():
                continue
            try:
                setattr(self.policy, name,
                        ort.InferenceSession(str(path), providers=heavy))
                self._moved.append(name)
            except Exception as e:      # a refused graph is a result, not a stop
                LOG_MOVE_FAILURES.append(f"{name}: {type(e).__name__}: {e}")

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
            "projectors": self.projectors,
            "projectors_moved_to_gpu": self._moved,
            "projector_move_failures": LOG_MOVE_FAILURES,
            "trt_opt_level": self.trt_opt_level,
            "trt_workspace_mb": self.trt_workspace_mb,
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
        # The four graphs that run once per denoising step. If moving these to the GPU
        # is worth doing, this is the number it has to beat.
        t["per_step_projectors"] = round(
            sum(self._sink.get(n, 0.0) for n in _PER_STEP_PROJECTORS), 3)
        t["decode_trt"] = round(self._sink.get("decode", 0.0), 3)
        # Whatever is left is numpy: masks, the Euler update, resize, tokenization.
        t["python_numpy"] = round(total - gpu - cpu, 3)
        return InferResult(np.asarray(chunk), t)
