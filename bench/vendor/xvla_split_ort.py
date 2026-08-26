# ─────────────────────────────────────────────────────────────────────────────
# VENDORED from spark-projects @ d342afa — orin-nano/xvla-runtime/xvla_runtime/split_ort.py
#
# The X-VLA split-engine runtime: twelve exported graphs run through ONNX Runtime
# with the TensorRT execution provider, denoising loop in numpy. Copied rather
# than imported so this benchmark repo stands alone and so the code that produced
# a number is pinned beside that number.
#
# ONLY MODIFICATION vs the original: the engine-prebuild subprocess imports this
# module as `bench.vendor.xvla_split_ort` from the repo root, instead of
# `xvla_runtime.split_ort` from the runtime package. The model code, provider
# stack, split layout and denoising loop are untouched.
#
# Note for anyone reading the loop: it is NOT Euler integration. X-VLA re-forms
# x_t by interpolating a FIXED noise draw against the current action estimate,
# and the transformer predicts the clean action directly. SmolVLA's
# `x_t += dt * v_t` produces plausible-looking garbage here.
#
# To re-sync: copy the file again, re-apply that one change, note the new sha.
# ─────────────────────────────────────────────────────────────────────────────
"""X-VLA split-engine inference on the Orin Nano (ONNX Runtime + TensorRT EP).

Mirrors the SmolVLA split runtime (`kaivuriprokkis/lerobot_vla/smolvla_split.py`): the
graphs are exported by `tools/export_split_onnx.py`, engines are prebuilt one per
subprocess, and the denoising loop lives in Python rather than in the graph.

The loop is NOT Euler integration. `XVLAModel.generate_actions` re-forms `x_t` by
interpolating between a FIXED noise draw and the current action estimate, and the
transformer predicts the clean action directly:

    x1 = randn(...); action = zeros_like(x1)
    for i in range(steps, 0, -1):
        t = i / steps
        x_t = x1 * t + action * (1 - t)
        action = transformer(action_with_noise=x_t, t=t, ...)

Getting this wrong (accumulating `x_t += dt * v_t` as SmolVLA does) still produces
plausible-looking actions, which is exactly why it is worth stating here.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

LOG = logging.getLogger(__name__)

_DEFAULT_CUDA_MEM_LIMIT = 3 << 30  # 3 GiB

# ImageNet statistics -- XVLAImageNetNormalizeProcessorStep, applied to [0,1] images
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def ep_context_path(split_dir: str | Path, onnx_file: str) -> Path:
    """Where the EPContext stand-in for `onnx_file` lives, if one has been dumped."""
    return Path(split_dir) / "ctx" / (Path(onnx_file).stem + "_ctx.onnx")


def make_session_options(disable_ort_fusions: bool = True) -> "ort.SessionOptions":
    """Session options for a TRT-EP session.

    ORT's own graph optimizer runs BEFORE provider partitioning and fuses patterns into
    `com.microsoft.*` contrib ops. TRT cannot take those, so they fall back — and after the
    FP16 weight cast the CPU EP has no `com.microsoft.Gelu` kernel for float16 at all, so a
    session that built fine in FP32 fails outright with NOT_IMPLEMENTED. Neither ONNX file
    contains a Gelu node; the fusion is created at load. Disabling ORT's fusions leaves
    plain ops that TRT can absorb, which is both the fix and generally what you want with
    the TRT EP doing its own optimization.
    """
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if disable_ort_fusions:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return so


def build_providers(cache_dir: str, precision: str = "fp16",
                    dump_ep_context_to: str | None = None) -> list:
    """TensorRT EP -> CUDA EP -> CPU EP, engine cache on disk.

    Same stack smolvla-runtime validated on this board.

    `dump_ep_context_to` asks the TRT EP to also write an **EPContext** model: a tiny ONNX
    that just points at the prebuilt engine. Loading that instead of the real graph means
    ORT never parses the multi-hundred-MB weight proto at session creation, which is the
    difference between the engines fitting alongside the control stack and not.
    """
    os.makedirs(cache_dir, exist_ok=True)
    trt_opts = {
        "device_id": 0,
        "trt_fp16_enable": precision == "fp16",
        "trt_bf16_enable": precision == "bf16",
        "trt_layer_norm_fp32_fallback": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache_dir,
        "trt_max_workspace_size": int(os.environ.get("TRT_WORKSPACE_MB", "512")) * (1 << 20),
        "trt_min_subgraph_size": 5,
    }
    # Lower optimization level explores fewer tactics -> smaller build peak.
    # Only affects the one-time build; a cached engine reloads identically.
    if os.environ.get("TRT_OPT_LEVEL"):
        trt_opts["trt_builder_optimization_level"] = int(os.environ["TRT_OPT_LEVEL"])
    if dump_ep_context_to:
        os.makedirs(dump_ep_context_to, exist_ok=True)
        trt_opts["trt_dump_ep_context_model"] = True
        trt_opts["trt_ep_context_file_path"] = dump_ep_context_to
        # embed_mode 0 keeps the engine in its own file and leaves the context ONNX
        # tiny; mode 1 would inline the engine and defeat the point.
        trt_opts["trt_ep_context_embed_mode"] = 0
    cuda_opts = {
        "device_id": 0,
        "gpu_mem_limit": _DEFAULT_CUDA_MEM_LIMIT,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "do_copy_in_default_stream": True,
    }
    providers: list = [("TensorrtExecutionProvider", trt_opts)]
    if not os.environ.get("TRT_DROP_CUDA_EP"):
        providers.append(("CUDAExecutionProvider", cuda_opts))
    providers.append("CPUExecutionProvider")
    return providers


# --------------------------------------------------------------------------------------
# preprocessing -- must match the lerobot pipeline exactly
# --------------------------------------------------------------------------------------


def resize_with_pad_chw(img_chw: np.ndarray, height: int, width: int,
                        pad_value: float = 0.0) -> np.ndarray:
    """lerobot `resize_with_pad`: keep aspect, bilinear, pad LEFT and TOP.

    Applied AFTER ImageNet normalization (that is the order `XVLAPolicy._prepare_images`
    inherits from the processor pipeline), so `pad_value=0.0` is zero in normalized
    space, not a black pixel.
    """
    import cv2

    c, h, w = img_chw.shape
    if (h, w) == (height, width):
        return img_chw

    ratio = max(w / width, h / height)
    rh, rw = int(h / ratio), int(w / ratio)
    resized = cv2.resize(
        img_chw.transpose(1, 2, 0), (rw, rh), interpolation=cv2.INTER_LINEAR
    )
    if resized.ndim == 2:
        resized = resized[:, :, None]
    canvas = np.full((height, width, c), pad_value, dtype=np.float32)
    canvas[height - rh :, width - rw :] = resized
    return canvas.transpose(2, 0, 1)


def preprocess_image(img_hwc_uint8: np.ndarray, size: int = 224) -> np.ndarray:
    """uint8 HxWx3 -> float32 [3,size,size], ImageNet-normalized then padded."""
    chw = img_hwc_uint8.transpose(2, 0, 1).astype(np.float32) / 255.0
    chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
    return resize_with_pad_chw(chw, size, size, pad_value=0.0)


def timestep_embedding(t: float, dim: int = 32, max_period: int = 100) -> np.ndarray:
    """`soft_transformer.timestep_embedding` for a scalar t -- cos first, then sin."""
    half = dim // 2
    freqs = np.exp(-math.log(max_period) * np.arange(half, dtype=np.float32) / half)
    args = t * freqs
    emb = np.concatenate([np.cos(args), np.sin(args)])
    if dim % 2:
        emb = np.concatenate([emb, np.zeros(1, dtype=np.float32)])
    return emb.astype(np.float32)[None]


# --------------------------------------------------------------------------------------
# engine prebuild
# --------------------------------------------------------------------------------------


def prebuild_engines(split_dir: str | Path, cache_dir: str, precision: str = "fp16",
                     only: list[str] | None = None, ep_context: bool = False) -> None:
    """Build each graph's TRT engine in its OWN subprocess.

    Not an optimization: during the SmolVLA work, two split engines building or resident
    in one process was enough to OOM 8 GB. A subprocess per graph returns all of the
    builder's memory to the OS between graphs.

    `ep_context` writes an EPContext stand-in per graph into `<split_dir>/ctx`, which the
    runtime prefers at load time. **Off by default**: `tools/memory_probe.py` showed the
    weights are resident once, not twice, so this saves load time rather than the RSS it
    was added for — and ORT currently mangles `trt_ep_context_file_path` against the
    engine-cache path into a nested directory it then fails to create.
    """
    split_dir = Path(split_dir)
    bundle = json.loads((split_dir / "bundle.json").read_text())
    ctx_dir = str(split_dir / "ctx") if ep_context else ""
    for graph in bundle["graphs"]:
        name = graph["name"]
        if only and name not in only:
            continue
        onnx_path = split_dir / graph["file"]
        LOG.info("prebuilding TRT engine for %s (subprocess)...", name)
        t0 = time.time()
        env = dict(
            os.environ,
            TRT_DROP_CUDA_EP="1",
            TRT_WORKSPACE_MB=os.environ.get("TRT_WORKSPACE_MB", "512"),
            TRT_OPT_LEVEL=os.environ.get("TRT_OPT_LEVEL", "2"),
        )
        proc = subprocess.run(
            [sys.executable, "-c", _BUILD_ONE_SRC, str(onnx_path), cache_dir, precision,
             ctx_dir],
            env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
            raise RuntimeError(
                f"engine build failed for {name} (exit {proc.returncode}):\n"
                + "\n".join(tail)
            )
        LOG.info("  %s built in %.0f s", name, time.time() - t0)


_BUILD_ONE_SRC = """
import sys
import numpy as np
import onnxruntime as ort
sys.path.insert(0, %r)
from bench.vendor.xvla_split_ort import build_providers

onnx_path, cache_dir, precision = sys.argv[1], sys.argv[2], sys.argv[3]
ctx_dir = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
from bench.vendor.xvla_split_ort import make_session_options
so = make_session_options()
sess = ort.InferenceSession(onnx_path, sess_options=so,
                            providers=build_providers(cache_dir, precision, ctx_dir))
# One real run so TRT builds every profile the graph needs, not just the first.
feed = {}
for inp in sess.get_inputs():
    dtype = np.int64 if "int64" in inp.type else np.float32
    feed[inp.name] = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=dtype)
sess.run(None, feed)
""" % str(Path(__file__).resolve().parents[2])


# --------------------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------------------


class XVLASplitPolicy:
    """Runs the exported X-VLA split graphs as one policy.

    Cold path (vision -> text_encoder -> cond) runs once per observation; the denoise
    graphs run `num_denoising_steps` times over the cached conditioning.
    """

    def __init__(self, split_dir: str | Path, cache_dir: str | None = None,
                 precision: str = "fp16", tokenizer_dir: str | None = None,
                 num_denoising_steps: int | None = None, seed: int | None = None):
        # Default the engine cache next to the graphs, not /tmp: twelve engines take
        # ~10 min to build and /tmp is cleared on reboot, which would mean paying that
        # every boot. (smolvla-runtime uses /tmp and does exactly that.)
        cache_dir = cache_dir or str(Path(split_dir) / "trt_cache")
        self.split_dir = Path(split_dir)
        self.bundle = json.loads((self.split_dir / "bundle.json").read_text())
        b = self.bundle

        self.num_views = b["num_image_views"]
        self.valid_views = b["valid_views"]
        self.tokens_per_view = b["tokens_per_view"]
        self.lang_len = b["lang_len"]
        self.chunk_size = b["chunk_size"]
        self.hidden = b["hidden_size"]
        self.dim_time = b["dim_time"]
        self.state_dim = b["max_state_dim"]
        self.steps = (num_denoising_steps if num_denoising_steps is not None
                      else b["num_denoising_steps"])
        if self.steps <= 0:
            raise ValueError("num_denoising_steps must be positive")
        self.rng = np.random.default_rng(seed)

        # Gripper channels get a sigmoid after the loop (BaseActionSpace.postprocess).
        # The pre-step zeroing is already baked into the denoise graph; this is only the
        # postprocess half, so it must match the checkpoint's action mode.
        gripper_by_mode = {"ee6d": (9, 19), "agibot_ee6d": (9, 19), "joint": (6, 13),
                           "so101_bimanual": (5, 11)}
        mode = b["action_mode"]
        if mode not in gripper_by_mode:
            raise ValueError(
                f"action_mode {mode!r} has no gripper-index mapping here; check "
                f"lerobot's action_hub.py and add it, or postprocess will be wrong"
            )
        self.gripper_idx = gripper_by_mode[mode]

        providers = build_providers(cache_dir, precision)
        so = make_session_options()

        def sess(name: str) -> ort.InferenceSession:
            spec = next(g for g in b["graphs"] if g["name"] == name)
            path = self.split_dir / spec["file"]
            # Prefer the EPContext stand-in when one exists: same engine, but ORT skips
            # parsing the full weight proto, which is most of the resident cost.
            ctx = ep_context_path(self.split_dir, spec["file"])
            if ctx.exists():
                path = ctx
            t0 = time.time()
            s = ort.InferenceSession(str(path), sess_options=so, providers=providers)
            LOG.info("loaded %s in %.1f s%s", name, time.time() - t0,
                     " (ep_context)" if path is ctx else "")
            return s

        def chain(prefix: str) -> list[ort.InferenceSession]:
            """Sessions for a split family, in execution order.

            Sorted by the numeric suffix, not lexicographically: `denoise_10` must not
            land between `denoise_1` and `denoise_2`.
            """
            names = [
                g["name"] for g in b["graphs"]
                if g["name"] == prefix or g["name"].startswith(prefix + "_")
            ]
            names.sort(key=lambda n: int(n.rsplit("_", 1)[1]) if n != prefix else 0)
            return [sess(n) for n in names]

        self.vision = chain("vision")
        self.text_encoder = chain("text_encoder")
        self.cond = sess("cond")
        self.denoise = chain("denoise")

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir or "facebook/bart-large")

        # Read the action width off the graph rather than assuming the mode's nominal
        # dim -- the exporter bakes shapes in, so the engine is the authority.
        x_t_spec = next(i for i in self.denoise[0].get_inputs() if i.name == "x_t")
        self.action_dim = int(x_t_spec.shape[-1])
        self.last_timings: dict[str, float] = {}

    # -- cold path ---------------------------------------------------------------------

    def encode_observation(self, images_hwc: list[np.ndarray], instruction: str) -> np.ndarray:
        """images -> cond_tokens [1, T_cond, hidden], run once per observation."""
        t0 = time.time()
        pixels = np.stack(
            [preprocess_image(img) for img in images_hwc[: self.valid_views]]
        ).astype(np.float32)
        img_feats = self.vision[0].run(None, {"pixel_values": pixels})[0]
        for s in self.vision[1:]:
            img_feats = s.run(None, {"hidden_in": img_feats})[0]
        self.last_timings["vision_ms"] = (time.time() - t0) * 1000

        # forward_vlm scatters valid views into a zero buffer: padded views read as
        # exactly zero rather than as the tower's response to a blank image.
        full = np.zeros((self.num_views, self.tokens_per_view, img_feats.shape[-1]),
                        dtype=np.float32)
        full[: len(img_feats)] = img_feats

        t0 = time.time()
        tok = self.tokenizer(
            instruction, max_length=self.lang_len, padding="max_length",
            truncation=True, padding_side="right", return_tensors="np",
        )
        vlm_features = self.text_encoder[0].run(
            None,
            {"input_ids": tok["input_ids"].astype(np.int64), "image_tokens": full[0:1]},
        )[0]
        for s in self.text_encoder[1:]:
            vlm_features = s.run(None, {"hidden_in": vlm_features})[0]
        self.last_timings["text_ms"] = (time.time() - t0) * 1000

        aux = full[1:].reshape(1, -1, full.shape[-1])
        t0 = time.time()
        cond_tokens = self.cond.run(
            None, {"vlm_features": vlm_features, "aux_visual": aux}
        )[0]
        self.last_timings["cond_ms"] = (time.time() - t0) * 1000
        return cond_tokens

    # -- hot path ----------------------------------------------------------------------

    def sample_actions(self, images_hwc: list[np.ndarray], instruction: str,
                       state: np.ndarray, x1: np.ndarray | None = None) -> np.ndarray:
        """`x1` overrides the noise draw -- needed to compare against the PyTorch
        reference, which draws its own inside `generate_actions`."""
        cond_tokens = self.encode_observation(images_hwc, instruction)

        proprio = np.zeros((1, self.state_dim), dtype=np.float32)
        flat = np.asarray(state, dtype=np.float32).ravel()[: self.state_dim]
        proprio[0, : len(flat)] = flat

        if x1 is None:
            x1 = self.rng.standard_normal(
                (1, self.chunk_size, self.action_dim)
            ).astype(np.float32)
        x1 = np.ascontiguousarray(x1, dtype=np.float32)
        action = np.zeros_like(x1)

        t0 = time.time()
        for i in range(self.steps, 0, -1):
            t = i / self.steps
            x_t = x1 * t + action * (1.0 - t)
            action = self._denoise_step(x_t, t, proprio, cond_tokens)
        self.last_timings["denoise_ms"] = (time.time() - t0) * 1000
        self.last_timings["steps"] = self.steps

        # EE6DActionSpace.postprocess: sigmoid the gripper logits
        action = action.copy()
        idx = list(self.gripper_idx)
        action[..., idx] = 1.0 / (1.0 + np.exp(-action[..., idx]))
        return action[0]

    def _denoise_step(self, x_t: np.ndarray, t: float, proprio: np.ndarray,
                      cond_tokens: np.ndarray) -> np.ndarray:
        feed = {
            "x_t": x_t.astype(np.float32),
            "t": np.array([t], dtype=np.float32),
            "proprio": proprio,
            "cond_tokens": cond_tokens,
        }
        out = self.denoise[0].run(None, feed)[0]
        for sess in self.denoise[1:]:
            out = sess.run(None, {"hidden_in": out})[0]
        return out
