"""Backend: X-VLA's twelve-graph split ONNX on ONNX Runtime + TensorRT EP.

Same idea as the SmolVLA split backend, a different shape of problem. SmolVLA needed
nine graphs; X-VLA needs twelve, and the reason is a measured build-memory curve rather
than a preference. TensorRT imports weights as FP32 working copies whatever the ONNX
dtype says, so the build peak tracks the weight slice one engine carries — on this
board, `build peak RSS ≈ 3.18 GB + 5.63 × (FP32 weight GB)`, leaving room for roughly
0.4 GB of weights (~100 M params) per engine. All three of X-VLA's heavy components
exceed that alone, so each is split further: DaViT + projector into 4, BART + token
embedding into 3, the policy transformer into 4, conditioning into 1.

Two structural differences from SmolVLA that this backend exists to measure

  **No KV cache is possible.** X-VLA's policy transformer is a bidirectional encoder
  over one concatenated sequence, so the conditioning tokens attend *to* the action
  tokens and change on every denoising step. All 24 blocks re-run over all 262 tokens,
  ten times. There is no prefill/decode seam to exploit; the only real lever on latency
  is `num_denoising_steps`.

  **The loop is not Euler integration.** It re-forms `x_t` by interpolating a fixed
  noise draw against the current action estimate and the transformer predicts the clean
  action directly. That is why `x1` — not a per-step noise sequence — is what gets
  injected for parity.

What is hoisted out of the loop is exact rather than approximate: the conditioning
projections, their positional-embedding slice and the soft prompts do not depend on
`x_t` or `t`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from .base import Backend, InferResult, tree_sha256
from .ort_split import _TimedSession


class OrtSplitXVLABackend(Backend):
    name = "ort-split-xvla"
    noise_injected = True

    def __init__(self, split_dir: Path, cache_dir: str | None = None,
                 precision: str = "fp16", num_steps: int | None = None,
                 tokenizer: str | None = None, seed: int = 0,
                 valid_views: int | None = None):
        self.split_dir = Path(split_dir)
        # Twelve engines are a long build to repeat and /tmp clears at boot, so the
        # cache defaults next to the graphs rather than under /tmp.
        self.cache_dir = cache_dir or str(self.split_dir / "trt_cache")
        self.precision = precision
        self.num_steps = num_steps
        if self.num_steps is not None and self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self.tokenizer = tokenizer
        self.seed = seed
        self.valid_views = valid_views
        self.policy = None
        self._sink: dict = {}

    def load(self) -> None:
        from ..vendor.xvla_split_ort import XVLASplitPolicy, prebuild_engines

        bundle = json.loads((self.split_dir / "bundle.json").read_text())
        self.bundle_valid_views = int(bundle["valid_views"])
        self.bundle_num_views = int(bundle["num_image_views"])
        if self.valid_views is None:
            self.valid_views = self.bundle_valid_views
        if self.valid_views != self.bundle_valid_views:
            raise ValueError(
                f"requested {self.valid_views} real view(s), but bundle declares "
                f"valid_views={self.bundle_valid_views}; use a separately exported bundle")
        tokenizer_source = self.tokenizer or "facebook/bart-large"
        self._tokenizer_name = str(tokenizer_source)
        tokenizer_path = Path(tokenizer_source)
        self._tokenizer_sha256 = (
            tree_sha256(tokenizer_path) if tokenizer_path.exists() else None)

        # One subprocess per graph. Not an optimization: two engines building or
        # resident in one process was enough to OOM 8 GB during the SmolVLA work.
        prebuild_engines(self.split_dir, self.cache_dir, self.precision)

        self.policy = XVLASplitPolicy(
            split_dir=self.split_dir, cache_dir=self.cache_dir,
            precision=self.precision, tokenizer_dir=self.tokenizer,
            num_denoising_steps=self.num_steps, seed=self.seed)

        # The split families are lists of sessions; label each by family + position so
        # the report shows which stage of a chain is expensive, not just the total.
        self._providers = {}
        for fam in ("vision", "text_encoder", "denoise"):
            chain = getattr(self.policy, fam)
            wrapped = []
            for i, s in enumerate(chain):
                self._providers[f"{fam}_{i}"] = s.get_providers()[0]
                wrapped.append(_TimedSession(s, f"{fam}_{i}", self._sink))
            setattr(self.policy, fam, wrapped)
        self._providers["cond"] = self.policy.cond.get_providers()[0]
        self.policy.cond = _TimedSession(self.policy.cond, "cond", self._sink)
        self._labels = list(self._providers)

    def artifact_paths(self) -> dict[str, Path]:
        return {"bundle": self.split_dir}

    def meta(self) -> dict:
        import onnxruntime as ort
        p = self.policy
        n_trt = sum(1 for v in self._providers.values()
                    if v == "TensorrtExecutionProvider")
        return {
            "backend": self.name,
            "family": "xvla",
            "split_dir": str(self.split_dir),
            "precision": self.precision,
            "num_steps": p.steps,
            "chunk_size": p.chunk_size,
            "action_dim": p.action_dim,
            "state_dim": p.state_dim,
            "num_views": p.num_views,
            "requested_views": self.valid_views,
            "bundle_valid_views": self.bundle_valid_views,
            "processed_views": p.valid_views,
            "valid_views": p.valid_views,
            "resize": [224, 224],
            "tokenizer": self._tokenizer_name,
            "tokenizer_sha256": self._tokenizer_sha256,
            "tokens_per_view": p.tokens_per_view,
            "lang_len": p.lang_len,
            "hidden_size": p.hidden,
            "action_mode": p.bundle.get("action_mode"),
            "n_graphs": len(self._providers),
            "n_graphs_on_trt": n_trt,
            "configured_provider_priority_per_graph": self._providers,
            "onnxruntime": ort.__version__,
            "engine_cache": self.cache_dir,
            "kv_cache": False,
            "kv_cache_note": "impossible — bidirectional policy transformer, "
                             "conditioning attends to action tokens and changes per step",
        }

    def infer(self, obs: Observation) -> InferResult:
        if len(obs.images) != self.bundle_valid_views:
            raise ValueError(
                f"observation has {len(obs.images)} view(s), bundle requires exactly "
                f"{self.bundle_valid_views}")
        self._sink.clear()
        t0 = time.perf_counter()
        # X-VLA injects x1, the single fixed noise draw the loop interpolates against,
        # not a per-step sequence — see the module docstring.
        x1 = obs.noise.astype(np.float32)
        chunk = self.policy.sample_actions(obs.images, obs.task, obs.state, x1=x1)
        total = (time.perf_counter() - t0) * 1000.0

        t = {"total": total}
        gpu = cpu = 0.0
        for label in self._labels:
            ms = self._sink.get(label, 0.0)
            t[f"graph.{label}"] = round(ms, 3)
            t[f"graph.{label}.calls"] = self._sink.get(label + "_n", 0)
            if self._providers.get(label) in ("TensorrtExecutionProvider",
                                              "CUDAExecutionProvider"):
                gpu += ms
            else:
                cpu += ms
        # The runtime's own coarse split, for cross-checking the wrapper's arithmetic.
        for k, v in (self.policy.last_timings or {}).items():
            t[f"runtime.{k}"] = round(float(v), 3)
        t["graphs_gpu"] = round(gpu, 3)
        t["graphs_cpu"] = round(cpu, 3)
        t["python_numpy"] = round(total - gpu - cpu, 3)
        return InferResult(np.asarray(chunk), t)
