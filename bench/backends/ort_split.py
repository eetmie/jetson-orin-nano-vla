"""Backend: the split 9-graph ONNX export on ONNX Runtime + TensorRT EP (FP16).

This is the incumbent — the path already validated on this board and already driving a
real machine (`kaivuriprokkis/lerobot_vla/smolvla_split.py`, vendored under
`bench/vendor/` so the measured code is pinned beside its numbers). It exists because the *monolithic* SmolVLA ONNX cannot TRT-build in
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

import math

from ..obs import Observation
from ..vendor.smolvla_split import (IMG_TOKENS, MAX_ACTION_DIM, MAX_STATE_DIM, VLM_DIM,
                                    make_att_2d_masks, resize_with_pad_uint8)
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
                 tokenizer: str | None = None, iobinding: bool = False):
        self.bundle = Path(bundle)
        self.cache_dir = cache_dir
        self.precision = precision
        self.action_dim = action_dim
        self.seed = seed
        self.projectors = projectors
        self.tokenizer = tokenizer
        self.iobinding = iobinding
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
        if self.iobinding:
            # Bind the KV cache to device once per inference instead of re-feeding
            # 72 MB of numpy per inference. Bit-identical; see split_iobind.py.
            from .split_iobind import enable_iobinding
            enable_iobinding(self.policy)
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

    def _sample_actions_multiview(self, obs: Observation) -> np.ndarray:
        """`sample_actions` for more than one REAL camera.

        The vendored runtime deploys a single-camera policy, so its `sample_actions`
        takes one image and fills every remaining slot with the padding embedding. That
        is not a limitation of the export — the reference base export carries two camera
        slots — so feeding a genuine second camera means rebuilding the prefix here
        rather than editing the file that runs on the robot.

        Everything numerical is the policy's own: the same vision session, the same
        language cache, the same mask construction, the same denoise loop. The only
        difference is that slot *i* gets a real vision pass instead of the cached
        padding embedding, and its 64 tokens are marked present rather than absent.

        This is also the measurement that matters for a bandwidth-limited board: a
        padded slot costs **zero** vision passes per inference (its embedding is
        computed once at load and reused) but still occupies its 64 tokens in the
        prefix. So the second camera's cost is one vision pass, not a longer sequence —
        the sequence was already that long.
        """
        p = self.policy
        n_real = len(obs.images)
        if n_real > p.n_cam_slots:
            raise ValueError(
                f"{n_real} cameras given but the export has {p.n_cam_slots} camera "
                f"slot(s) (prefix {p.prefix_len}). Re-export with more slots, or run "
                f"with --views {p.n_cam_slots}.")

        img_embs = [p._run_vision(resize_with_pad_uint8(im)) for im in obs.images]
        lang_emb, lang_mask = p._embed_language(obs.task)

        s_norm = p.norm.normalize_state(
            np.asarray(obs.state, dtype=np.float32).reshape(-1))
        s_pad = np.zeros((1, MAX_STATE_DIM), dtype=np.float32)
        s_pad[0, :s_norm.shape[0]] = s_norm
        state_emb = p._run_single(p.state_proj, s_pad).reshape(1, 1, VLM_DIM)

        n_pad_cams = p.n_cam_slots - n_real
        embs = np.concatenate(
            img_embs + [p._pad_cam_emb] * n_pad_cams + [lang_emb, state_emb],
            axis=1).astype(np.float32)
        pad_masks = np.concatenate(
            [np.ones((1, IMG_TOKENS), dtype=bool)] * n_real
            + [np.zeros((1, IMG_TOKENS), dtype=bool)] * n_pad_cams
            + [lang_mask, np.ones((1, 1), dtype=bool)], axis=1)
        att_masks = np.zeros((1, p.prefix_len), dtype=bool)
        att_masks[0, -1] = True                       # state starts a new block

        kv = p.prefill.run(p._prefill_kv_names, {
            "attention_mask": make_att_2d_masks(pad_masks, att_masks),
            "position_ids": (np.cumsum(pad_masks, axis=1) - 1).astype(np.int64),
            "vlm_embeds": embs,
        })

        x_t = obs.noise.astype(np.float32).copy()
        dt = -1.0 / p.num_steps
        t = 1.0
        prefix_pad_2d = np.broadcast_to(
            pad_masks[:, None, :], (1, p.chunk_size, p.prefix_len))
        suffix = np.ones((1, p.chunk_size), dtype=bool)
        full_att_2d = np.concatenate(
            [prefix_pad_2d, make_att_2d_masks(suffix, suffix)], axis=2)
        pos_ids = (pad_masks.sum(axis=-1, keepdims=True)
                   + np.cumsum(suffix, axis=1) - 1).astype(np.int64)
        kv_feed = {}
        for i in range(p.n_layers):
            kv_feed[f"past_key_{i}"] = kv[2 * i]
            kv_feed[f"past_value_{i}"] = kv[2 * i + 1]

        while t >= -dt / 2:
            x_t = x_t + dt * p._denoise_step(x_t, t, full_att_2d, pos_ids, kv_feed)
            t += dt
        return p.norm.unnormalize_action(x_t[0, :, :self.action_dim])

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
        if len(obs.images) > 1:
            chunk = self._sample_actions_multiview(obs)
        else:
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
