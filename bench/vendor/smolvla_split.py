# ─────────────────────────────────────────────────────────────────────────────
# VENDORED from kaivuriprokkis @ df54f3c — lerobot_vla/smolvla_split.py
#
# This is the split-engine SmolVLA runtime that actually drives the MASI
# excavator: the 9-graph ONNX export run through ONNX Runtime with the TensorRT
# execution provider, flow-matching denoise loop in numpy. It is copied here
# rather than imported so this benchmark repo stands alone, and so the exact code
# that produced a number is pinned next to that number.
#
# ONLY MODIFICATION vs the original: the engine-prebuild subprocess invokes this
# module as `bench.vendor.smolvla_split` instead of `lerobot_vla.smolvla_split`
# (two lines in prebuild_engines). The model code, provider stack, precision
# settings and denoise loop are untouched — changing them would mean
# benchmarking something other than what runs on the machine.
#
# To re-sync: copy the file again, re-apply that one change, note the new sha.
# ─────────────────────────────────────────────────────────────────────────────
"""SmolVLA split-engine inference — ONNX Runtime + TensorRT EP, numpy only.

Runs the 9-graph split export (see spark-projects/orin-nano/smolvla-runtime/
exports/ainekko_base_split) with the flow-matching denoise loop in Python:

    vision (x per camera) + text + state_proj  ->  prefix embeddings [1,177,960]
    expert_prefill (once)                      ->  KV cache
    for each of num_steps:  action_in + time MLP -> expert_decode -> action_out
                            x_t += dt * v_t

This is why it fits the Orin Nano's 8 GB: each engine carries only its slice
of the 450M weights, so the TRT builds peak at ~2-3 GB instead of the ~7 GB
that killed the monolithic export (notes/findings.md, 2026-06-16/17).

Orchestration mirrors github.com/aifoundry-org/ETARS modeling_smolvla_ort.py,
checked against lerobot 0.5.1 modeling_smolvla.py. The prefix length is read
from the prefill graph at load time — it depends on how many camera slots the
bundle was exported with (ainekko base: 2 slots -> 177 = 2x64 + 48 lang + 1
state; our single-camera excavator export: 1 slot -> 113). Chunk 50; padded
action dim 32.

Heavy graphs (vision / prefill / decode) run on the TensorRT EP with an
on-disk engine cache; tiny projectors and the text embedding run on CPU.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import numpy as np

LOG = logging.getLogger("smolvla_split")

# Static dims of the split export (verified from the ONNX graphs).
VLM_DIM = 960
EXPERT_DIM = 720
IMG_TOKENS = 64
# Camera slots (and with them the prefix length) are baked into the prefill
# graph at export time and DIFFER between bundles: the ainekko base export has
# 2 slots (prefix 64+64+48+1 = 177), our fine-tuned excavator export has 1
# (64+48+1 = 113). The real value is read from the graph in __init__;
# this is only the fallback when the graph axis is dynamic.
DEFAULT_NUM_CAM_SLOTS = 2
LANG_LEN = 48
# The action chunk length is baked into the expert-decode graph at export time and
# DIFFERS between bundles: lerobot/smolvla_base and every excavator export so far use
# 50, but a bundle trained with --policy.chunk_size=12 or 30 carries that instead. The
# real value is read from the graph in __init__; this is only the fallback when the
# graph axis is dynamic. Feeding a 50-step suffix to a 12-step graph is a shape error,
# and feeding 12 to a 50-step graph would silently under-drive the action head.
DEFAULT_CHUNK_SIZE = 50
MAX_ACTION_DIM = 32
MAX_STATE_DIM = 32
IMG_SIZE = 512
MIN_PERIOD = 4e-3
MAX_PERIOD = 4.0

_DEFAULT_TRT_WORKSPACE = 1 << 30   # 1 GiB
_DEFAULT_CUDA_MEM_LIMIT = 3 << 30  # 3 GiB


def build_providers(cache_dir: str, precision: str = "fp16"):
    """TensorRT EP -> CUDA EP -> CPU EP, engine cache on disk.

    Same stack smolvla-runtime validated on this board (backends/ort.py).
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
        "trt_max_workspace_size": int(os.environ.get("TRT_WORKSPACE_MB", "1024")) * (1 << 20),
        "trt_min_subgraph_size": 5,
    }
    # Lower optimization level explores fewer tactics -> smaller build peak.
    # Only affects the one-time build; a cached engine reloads identically.
    if os.environ.get("TRT_OPT_LEVEL"):
        trt_opts["trt_builder_optimization_level"] = int(os.environ["TRT_OPT_LEVEL"])
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


def make_att_2d_masks(pad_masks: np.ndarray, att_masks: np.ndarray) -> np.ndarray:
    """SmolVLA's blockwise-causal 2D attention mask from 1D pad + att masks."""
    cumsum = np.cumsum(att_masks, axis=1)
    att_2d = cumsum[:, np.newaxis, :] <= cumsum[:, :, np.newaxis]
    pad_2d = pad_masks[:, np.newaxis, :] & pad_masks[:, :, np.newaxis]
    return att_2d & pad_2d


def sinusoidal_time_embedding(t: float, dim: int = EXPERT_DIM) -> np.ndarray:
    fraction = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = MIN_PERIOD * (MAX_PERIOD / MIN_PERIOD) ** fraction
    sin_input = (2 * math.pi / period) * t
    return np.concatenate([np.sin(sin_input), np.cos(sin_input)]).astype(np.float32)


def resize_with_pad_uint8(img_hwc: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """uint8 HxWx3 -> float32 [1,3,size,size] in [-1,1].

    Matches lerobot's resize_with_pad: keep aspect, bilinear, pad LEFT and TOP
    with 0 (in [0,1] space, i.e. -1 after the SigLIP [-1,1] normalization).
    """
    import cv2

    h, w = img_hwc.shape[:2]
    ratio = max(w / size, h / size)
    rw, rh = int(w / ratio), int(h / ratio)
    resized = cv2.resize(img_hwc, (rw, rh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[size - rh:, size - rw:] = resized          # pad top + left
    chw = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    return (chw * 2.0 - 1.0)[None]


HEAVY_GRAPHS = ("smolvlm_vision.onnx", "smolvlm_expert_prefill.onnx",
                "smolvlm_expert_decode.onnx")


def prebuild_engines(split_dir: str | Path, cache_dir: str,
                     precision: str = "fp16") -> None:
    """Build + cache the heavy TRT engines, ONE SUBPROCESS PER GRAPH.

    The builds cannot share a process on 8 GB: a build peaks at ~2-3 GB of
    scratch, and every already-created session keeps its engine + weights
    resident — building vision then prefill in one process OOMs exactly the
    way the monolith did. A subprocess per graph returns the memory between
    builds; afterwards the main process only ever loads from cache.
    """
    import subprocess
    import sys as _sys

    cache = Path(cache_dir)
    if len(list(cache.glob("*.engine"))) >= len(HEAVY_GRAPHS):
        return
    for name in HEAVY_GRAPHS:
        LOG.info("prebuilding TRT engine for %s (subprocess, ~1 min)...", name)
        t0 = time.perf_counter()
        # Build-peak settings proven to fit 8 GB on this board: no CUDA EP,
        # 512 MB workspace, optimization level 2. Only the one-time build is
        # affected — cached engines reload identically (and load with the
        # full provider stack later).
        env = dict(os.environ, TRT_DROP_CUDA_EP="1")
        env.setdefault("TRT_WORKSPACE_MB", "512")
        env.setdefault("TRT_OPT_LEVEL", "2")
        r = subprocess.run(
            [_sys.executable, "-m", "bench.vendor.smolvla_split",
             "--build-one", str(Path(split_dir) / name),
             "--cache-dir", str(cache_dir), "--precision", precision],
            env=env, cwd=str(Path(__file__).resolve().parents[2]),
        )
        if r.returncode != 0:
            raise RuntimeError(f"engine build failed for {name}")
        LOG.info("  built %s in %.0fs", name, time.perf_counter() - t0)


class NormStats:
    """MEAN_STD normalization from a LeRobot stats dict (or identity)."""

    def __init__(self, state_mean=None, state_std=None,
                 action_mean=None, action_std=None):
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std

    @classmethod
    def from_lerobot_stats(cls, stats: dict, state_key="observation.state",
                           action_key="action"):
        def get(key, field):
            v = stats.get(key, {}).get(field)
            return None if v is None else np.asarray(v, dtype=np.float32).reshape(-1)
        return cls(get(state_key, "mean"), get(state_key, "std"),
                   get(action_key, "mean"), get(action_key, "std"))

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_mean is None:
            return state
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_mean is None:
            return action
        return action * (self.action_std + 1e-8) + self.action_mean


class SmolVLASplitPolicy:
    """The 9-graph split SmolVLA policy with the denoise loop in Python."""

    def __init__(self,
                 split_dir: str | Path,
                 tokenizer_dir: str | Path,
                 cache_dir: str = "/tmp/smolvla_split_cache",
                 precision: str = "fp16",
                 num_steps: int = 10,
                 action_dim: int = 4,
                 norm: NormStats | None = None,
                 seed: int | None = None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        split_dir = Path(split_dir)
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self.num_steps = num_steps
        self.action_dim = action_dim
        self.norm = norm or NormStats()
        self._rng = np.random.default_rng(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

        prebuild_engines(split_dir, cache_dir, precision)

        heavy = build_providers(cache_dir, precision=precision)
        cpu = ["CPUExecutionProvider"]

        def sess(name, providers):
            t0 = time.perf_counter()
            s = ort.InferenceSession(str(split_dir / name), providers=providers)
            LOG.info("session %-28s %-26s %.1fs", name, s.get_providers()[0],
                     time.perf_counter() - t0)
            return s

        LOG.info("Loading split sessions (first run builds TRT engines — minutes)...")
        self.vision = sess("smolvlm_vision.onnx", heavy)
        self.prefill = sess("smolvlm_expert_prefill.onnx", heavy)
        self.decode = sess("smolvlm_expert_decode.onnx", heavy)
        self.text = sess("smolvlm_text.onnx", cpu)
        self.state_proj = sess("state_projector.onnx", cpu)
        self.action_in = sess("action_in_projector.onnx", cpu)
        self.action_out = sess("action_out_projector.onnx", cpu)
        self.time_in = sess("time_in_projector.onnx", cpu)
        self.time_out = sess("time_out_projector.onnx", cpu)

        self.n_layers = sum(1 for i in self.decode.get_inputs()
                            if i.name.startswith("past_key_"))
        LOG.info("decode expects %d KV layers", self.n_layers)

        # Ask the prefill for its KV outputs BY NAME. Output order differs
        # between bundles (ainekko base emits a leading vlm_output_embeds, our
        # excavator export emits only the 32 KV tensors); names are stable.
        self._prefill_kv_names = [
            name for i in range(self.n_layers)
            for name in (f"present_key_{i}", f"present_value_{i}")]
        have = {o.name for o in self.prefill.get_outputs()}
        missing = [n for n in self._prefill_kv_names if n not in have]
        if missing:
            raise ValueError(f"prefill graph lacks expected KV outputs: {missing[:4]} ...")

        # Read the prefix length the prefill graph was exported with (see the
        # note at DEFAULT_NUM_CAM_SLOTS) and derive the camera-slot count.
        dim = next((i.shape[1] for i in self.prefill.get_inputs()
                    if i.name == "position_ids"), None)
        if isinstance(dim, int):
            n_cams, rem = divmod(dim - LANG_LEN - 1, IMG_TOKENS)
            if rem or n_cams < 1:
                raise ValueError(
                    f"prefill prefix length {dim} does not decompose as "
                    f"n*{IMG_TOKENS} + {LANG_LEN} + 1 — unknown export layout")
            self.n_cam_slots, self.prefix_len = n_cams, dim
        else:  # dynamic axis: keep the historical layout
            self.n_cam_slots = DEFAULT_NUM_CAM_SLOTS
            self.prefix_len = self.n_cam_slots * IMG_TOKENS + LANG_LEN + 1
        LOG.info("prefill expects prefix %d (%d camera slot(s))",
                 self.prefix_len, self.n_cam_slots)

        # Same treatment for the action chunk length: the decode graph was exported with
        # position_ids [1, chunk] and attention_mask [1, chunk, prefix + chunk], so the
        # value is recoverable and cross-checkable rather than assumed.
        dim = next((i.shape[1] for i in self.decode.get_inputs()
                    if i.name == "position_ids"), None)
        if isinstance(dim, int):
            self.chunk_size = dim
            total = next((i.shape[2] for i in self.decode.get_inputs()
                          if i.name == "attention_mask"), None)
            if isinstance(total, int) and total != self.prefix_len + self.chunk_size:
                raise ValueError(
                    f"decode graph is inconsistent: attention_mask spans {total} but "
                    f"prefix {self.prefix_len} + chunk {self.chunk_size} = "
                    f"{self.prefix_len + self.chunk_size} — mismatched export")
        else:  # dynamic axis: keep the historical layout
            self.chunk_size = DEFAULT_CHUNK_SIZE
        LOG.info("decode expects a %d-step action chunk", self.chunk_size)

        # The empty-camera slot embedding is a constant (all -1 image, the
        # lerobot padding convention) — compute once, reuse every step.
        self._pad_cam_emb = None
        if self.n_cam_slots > 1:
            pad_img = -np.ones((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            self._pad_cam_emb = self._run_vision(pad_img)

        # Language is fixed per instruction in practice — cache by string.
        self._lang_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ── component wrappers ───────────────────────────────────────────────────

    @staticmethod
    def _run_single(sess, value: np.ndarray) -> np.ndarray:
        """Run a single-input graph, feeding by the graph's DECLARED input name.

        Tensor names differ between export generations (ainekko base: 'time' /
        'action'; our exporter: 'action_time' / 'expert_out' / 'hidden'), but
        every one of these graphs takes exactly one input of identical shape,
        so binding by declared name works for all bundles.
        """
        return sess.run(None, {sess.get_inputs()[0].name: value})[0]

    def _run_vision(self, img_bchw: np.ndarray) -> np.ndarray:
        emb = self._run_single(self.vision, img_bchw)            # [1,64,960]
        return emb * math.sqrt(emb.shape[-1])

    def _embed_language(self, instruction: str) -> tuple[np.ndarray, np.ndarray]:
        if instruction in self._lang_cache:
            return self._lang_cache[instruction]
        task = instruction if instruction.endswith("\n") else instruction + "\n"
        tok = self.tokenizer(task, padding="max_length", padding_side="right",
                             max_length=LANG_LEN, truncation=True,
                             return_tensors="np")
        tokens = tok["input_ids"].astype(np.int64)               # [1,48]
        mask = tok["attention_mask"].astype(bool)                # [1,48]
        emb = self._run_single(self.text, tokens)               # [1,48,960]
        emb = emb * math.sqrt(emb.shape[-1])
        self._lang_cache[instruction] = (emb, mask)
        return emb, mask

    # ── main inference ───────────────────────────────────────────────────────

    def sample_actions(self, image_hwc_uint8: np.ndarray, instruction: str,
                       state: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
        """One observation -> (chunk_size, action_dim) unnormalized action chunk."""
        # prefix: [cam1(64), pad-cam(64) x (slots-1), lang(48), state(1)]
        img = resize_with_pad_uint8(image_hwc_uint8)
        img_emb = self._run_vision(img)                          # [1,64,960]
        lang_emb, lang_mask = self._embed_language(instruction)

        s = self.norm.normalize_state(np.asarray(state, dtype=np.float32).reshape(-1))
        s_pad = np.zeros((1, MAX_STATE_DIM), dtype=np.float32)
        s_pad[0, :s.shape[0]] = s
        state_emb = self._run_single(self.state_proj, s_pad).reshape(1, 1, VLM_DIM)

        n_pad_cams = self.n_cam_slots - 1
        embs = np.concatenate(
            [img_emb] + [self._pad_cam_emb] * n_pad_cams + [lang_emb, state_emb],
            axis=1,
        ).astype(np.float32)                                     # [1,prefix,960]
        pad_masks = np.concatenate(
            [np.ones((1, IMG_TOKENS), dtype=bool)]               # real camera
            + [np.zeros((1, IMG_TOKENS), dtype=bool)] * n_pad_cams  # empty slots
            + [lang_mask,
               np.ones((1, 1), dtype=bool)],                     # state token
            axis=1)                                              # [1,prefix]
        att_masks = np.zeros((1, self.prefix_len), dtype=bool)
        att_masks[0, -1] = True                                  # state starts a new block

        prefix_att_2d = make_att_2d_masks(pad_masks, att_masks)  # [1,177,177]
        prefix_pos = (np.cumsum(pad_masks, axis=1) - 1).astype(np.int64)

        kv = self.prefill.run(self._prefill_kv_names, {
            "attention_mask": prefix_att_2d,
            "position_ids": prefix_pos,
            "vlm_embeds": embs,
        })

        # denoise loop
        if noise is None:
            noise = self._rng.standard_normal(
                (1, self.chunk_size, MAX_ACTION_DIM)).astype(np.float32)
        x_t = noise.copy()
        dt = -1.0 / self.num_steps
        t = 1.0

        # constant across steps
        prefix_pad_2d = np.broadcast_to(
            pad_masks[:, None, :], (1, self.chunk_size, self.prefix_len))
        suffix_pad = np.ones((1, self.chunk_size), dtype=bool)
        suffix_att = np.ones((1, self.chunk_size), dtype=bool)        # action block: causal-in-block
        suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_att_2d = np.concatenate(
            [prefix_pad_2d, suffix_att_2d], axis=2)              # [1,50,227]
        pos_ids = (pad_masks.sum(axis=-1, keepdims=True)
                   + np.cumsum(suffix_pad, axis=1) - 1).astype(np.int64)

        kv_feed = {}
        for i in range(self.n_layers):
            kv_feed[f"past_key_{i}"] = kv[2 * i]
            kv_feed[f"past_value_{i}"] = kv[2 * i + 1]

        while t >= -dt / 2:
            v_t = self._denoise_step(x_t, t, full_att_2d, pos_ids, kv_feed)
            x_t += dt * v_t
            t += dt

        actions = x_t[0, :, :self.action_dim]                    # (50, act_dim)
        return self.norm.unnormalize_action(actions)

    def _denoise_step(self, x_t, t, full_att_2d, pos_ids, kv_feed) -> np.ndarray:
        action_emb = self._run_single(self.action_in, x_t)               # [1,50,720]
        time_emb = np.broadcast_to(
            sinusoidal_time_embedding(t)[None, None, :], action_emb.shape)
        ate = np.concatenate([action_emb, time_emb], axis=2).astype(np.float32)
        ate = self._run_single(self.time_in, ate)
        ate = ate * (1.0 / (1.0 + np.exp(-ate)))                         # SiLU
        suffix_embs = self._run_single(self.time_out, ate)               # [1,50,720]

        feeds = {
            "attention_mask": full_att_2d,
            "position_ids": pos_ids,
            "expert_embeds": suffix_embs,
            **kv_feed,
        }
        expert_out = self.decode.run(None, feeds)[0]  # single output in every bundle
        return self._run_single(self.action_out, expert_out.astype(np.float32))


def _build_one(onnx_path: str, cache_dir: str, precision: str) -> None:
    """Subprocess entry: create one TRT-EP session so its engine gets cached."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path,
                                providers=build_providers(cache_dir, precision))
    if sess.get_providers()[0] != "TensorrtExecutionProvider":
        raise SystemExit(f"TRT EP did not register for {onnx_path}: "
                         f"{sess.get_providers()}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--build-one", required=True)
    ap.add_argument("--cache-dir", default="/tmp/smolvla_split_cache")
    ap.add_argument("--precision", default="fp16")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    _build_one(a.build_one, a.cache_dir, a.precision)
