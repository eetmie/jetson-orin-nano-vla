"""Backend: a MONOLITHIC ONNX through ONNX Runtime — the A/B the split path exists for.

The whole model as one graph, denoise loop unrolled, run on the TensorRT EP (or on the
CUDA EP with `--no-trt`). Nobody publishes an honest number for this on an 8 GB Orin,
which is exactly why it is worth taking: it is the counterfactual that says whether the
nine-graph split is an optimization or a necessity.

The prior, all measured on this class of board:

  * the 10-step SmolVLA export is 108,695 nodes and did **not** TRT-build in 8 GB —
    not at FP16, not at FP32, not headless, not with aggressive builder knobs. It
    either hard-OOM'd or, after ~85 minutes of tactic-skipping, failed outright with
    `Error Code 10: Could not find any implementation for node`.
  * cutting to 5 steps (61,370 nodes) barely moved it: ~6.7 GB peak against ~7.4 GB.
    The floor is node-count independent — TensorRT imports all 450 M weights as FP32
    working copies before it optimizes anything.
  * on the CUDA EP, with no engine build to fail, the same monolith ran: **532 ms mean,
    498 ms p50** — finite, plausible output, ~2 Hz.

So there are three outcomes worth distinguishing, and this backend reports which one
happened rather than only how fast it was:

  TRT built        the split path is an optimization, not a necessity
  CUDA fallback    the ~500 ms regime; a real deploy option for a chunked policy
  CPU fallback     a wrong number waiting to be quoted as a right one

That last case is the reason this file checks providers explicitly. When TensorRT gives
up, ONNX Runtime does not error — it partitions the graph elsewhere and keeps going. A
silent CPU fallback still returns a finite, plausible action chunk. `active_provider`
and `trt_engine_cached` go into every result so the number carries its own caveat.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from ..vendor.smolvla_split import (LANG_LEN, MAX_STATE_DIM, resize_with_pad_uint8)
from .base import Backend, InferResult, load_export_info, load_stats

#: The monolithic SmolVLA export has been through more than one naming generation
#: (`image` vs `image0`, `image_mask` vs `img_mask0`). Resolve by role, not by a
#: hardcoded name, so an older or newer export still binds.
_ROLE_PATTERNS = {
    "image": ("image0", "image", "images", "pixel_values"),
    "image_mask": ("img_mask0", "image_mask", "img_masks", "image_masks"),
    "lang_tokens": ("lang_tokens", "input_ids", "tokens"),
    "lang_masks": ("lang_masks", "attention_mask", "lang_mask"),
    "state": ("state", "proprio", "observation_state"),
    "noise": ("noise", "x1", "x_t"),
}


def _resolve_roles(sess) -> dict[str, str]:
    names = [i.name for i in sess.get_inputs()]
    lowered = {n.lower(): n for n in names}
    roles: dict[str, str] = {}
    for role, candidates in _ROLE_PATTERNS.items():
        for c in candidates:
            if c in lowered and lowered[c] not in roles.values():
                roles[role] = lowered[c]
                break
    return roles


class OrtMonoBackend(Backend):
    name = "ort-mono"
    noise_injected = True

    def __init__(self, onnx_path: Path, cache_dir: str, precision: str = "fp16",
                 use_trt: bool = True, bundle: Path | None = None,
                 tokenizer: str | None = None, action_dim: int | None = None,
                 drop_cuda_ep: bool = False, trt_opt_level: int | None = None,
                 trt_workspace_mb: int | None = None):
        self.onnx_path = Path(onnx_path)
        self.cache_dir = cache_dir
        self.precision = precision
        self.use_trt = use_trt
        self.bundle = Path(bundle) if bundle else None
        self.tokenizer_id = tokenizer
        self.action_dim = action_dim
        self.drop_cuda_ep = drop_cuda_ep
        self.trt_opt_level = trt_opt_level
        self.trt_workspace_mb = trt_workspace_mb
        self._info = load_export_info(self.bundle) if self.bundle else {}
        self.sess = None
        self._build_s: float | None = None

    def load(self) -> None:
        import os
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from ..vendor.smolvla_split import build_providers

        if self.trt_opt_level is not None:
            os.environ["TRT_OPT_LEVEL"] = str(self.trt_opt_level)
        if self.trt_workspace_mb is not None:
            os.environ["TRT_WORKSPACE_MB"] = str(self.trt_workspace_mb)
        if self.drop_cuda_ep:
            os.environ["TRT_DROP_CUDA_EP"] = "1"

        if self.use_trt:
            providers = build_providers(self.cache_dir, precision=self.precision)
        else:
            providers = [("CUDAExecutionProvider",
                          {"device_id": 0, "arena_extend_strategy": "kNextPowerOfTwo"}),
                         "CPUExecutionProvider"]

        # Session creation on a 108k-node graph is itself minutes of graph
        # optimization, before TensorRT is even asked to build anything.
        t0 = time.perf_counter()
        self.sess = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self._build_s = time.perf_counter() - t0

        self.roles = _resolve_roles(self.sess)
        missing = [r for r in ("image", "lang_tokens", "state") if r not in self.roles]
        if missing:
            raise RuntimeError(
                f"could not bind {missing} on {self.onnx_path.name}. Graph inputs are: "
                f"{[i.name for i in self.sess.get_inputs()]} — add the names to "
                f"_ROLE_PATTERNS in bench/backends/ort_mono.py")

        tok = self.tokenizer_id or (str(self.bundle / "tokenizer") if self.bundle else None)
        self.tokenizer = AutoTokenizer.from_pretrained(tok) if tok else None
        self.norm = load_stats(self.bundle) if self.bundle else load_stats(Path("."))
        self._lang_cache: dict[str, tuple] = {}
        self._shapes = {i.name: i.shape for i in self.sess.get_inputs()}

    def meta(self) -> dict:
        import onnxruntime as ort
        active = self.sess.get_providers()
        engines = list(Path(self.cache_dir).glob("*.engine")) if self.use_trt else []
        return {
            "backend": self.name,
            "onnx": str(self.onnx_path),
            "onnx_size_mb": round(self.onnx_path.stat().st_size / 1e6, 1),
            "requested_trt": self.use_trt,
            "precision": self.precision,
            # The three fields that decide whether this number means anything.
            "active_provider": active[0] if active else None,
            "active_providers": active,
            "trt_engine_cached": bool(engines),
            "n_cached_engines": len(engines),
            "ran_on_gpu": bool(active and active[0] != "CPUExecutionProvider"),
            "session_create_s": round(self._build_s or 0, 1),
            "input_roles": self.roles,
            "graph_inputs": {i.name: list(i.shape) for i in self.sess.get_inputs()},
            "graph_outputs": [o.name for o in self.sess.get_outputs()],
            "onnxruntime": ort.__version__,
            "engine_cache": self.cache_dir,
            "trt_opt_level": self.trt_opt_level,
            "trt_workspace_mb": self.trt_workspace_mb,
            "export_info": self._info,
            "chunk_size": self._info.get("chunk_size"),
        }

    def _language(self, task: str):
        if task not in self._lang_cache:
            if self.tokenizer is None:
                ids = np.ones((1, LANG_LEN), dtype=np.int64)
                mask = np.ones((1, LANG_LEN), dtype=bool)
            else:
                text = task if task.endswith("\n") else task + "\n"
                tok = self.tokenizer(text, padding="max_length", padding_side="right",
                                     max_length=LANG_LEN, truncation=True,
                                     return_tensors="np")
                ids = tok["input_ids"].astype(np.int64)
                mask = tok["attention_mask"].astype(bool)
            self._lang_cache[task] = (ids, mask)
        return self._lang_cache[task]

    def infer(self, obs: Observation) -> InferResult:
        t0 = time.perf_counter()
        img = resize_with_pad_uint8(obs.image)
        s = self.norm.normalize_state(np.asarray(obs.state, dtype=np.float32).reshape(-1))
        s_pad = np.zeros((1, MAX_STATE_DIM), dtype=np.float32)
        s_pad[0, :s.shape[0]] = s
        ids, mask = self._language(obs.task)

        feed: dict = {self.roles["image"]: img,
                      self.roles["lang_tokens"]: ids,
                      self.roles["state"]: s_pad}
        if "image_mask" in self.roles:
            feed[self.roles["image_mask"]] = np.ones((1,), dtype=bool)
        if "lang_masks" in self.roles:
            feed[self.roles["lang_masks"]] = mask
        if "noise" in self.roles:
            shape = self._shapes[self.roles["noise"]]
            n = obs.noise
            if all(isinstance(d, int) for d in shape) and tuple(shape) != n.shape:
                n = n[:, :shape[1], :shape[2]]
            feed[self.roles["noise"]] = n.astype(np.float32)
        t_pre = time.perf_counter()

        out = self.sess.run(None, feed)[0]
        t_model = time.perf_counter()

        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if self.action_dim:
            arr = arr[:, :self.action_dim]
        chunk = self.norm.unnormalize_action(arr)
        t1 = time.perf_counter()

        return InferResult(chunk, {
            "total": (t1 - t0) * 1000.0,
            "preprocess": (t_pre - t0) * 1000.0,
            "model": (t_model - t_pre) * 1000.0,
            "postprocess": (t1 - t_model) * 1000.0,
        })
