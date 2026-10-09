# ─────────────────────────────────────────────────────────────────────────────
# Written for this repo (not vendored). The host loop follows smolvla_split.py and the
# multi-camera prefix of backends/ort_split.py, which are the tested ORT implementation;
# the engines, serial builds and fixture comparison are GR00T's (groot_trt.py).
# ─────────────────────────────────────────────────────────────────────────────
"""SmolVLA split-engine inference on the TensorRT runtime, numpy host glue.

The bundle's graphs, minus `smolvlm_text` (a single Gather): its table ships as
`embed_tokens.npy` and is memory-mapped, so only the prompt's rows are ever read.

Per observation (V real cameras of `n_cam_slots`, `num_steps` Euler steps):

    vision        [1,3,512,512] -> [1,64,960] per real camera; an empty slot reuses the
                  embedding of the all -1 image, computed once
    host          prefix = cameras + prompt rows + state_proj(state), masks, positions
    prefill       -> 16 layers of K/V, kept for the whole loop
    num_steps x:  action_in, sinusoidal time, time_in, SiLU, time_out, decode,
                  action_out -> v_t;  x_t += dt * v_t
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .groot_trt import FIXTURE_MAX_PCT_RANGE, FIXTURE_MIN_COSINE, _cmp
from .smolvla_split import (IMG_TOKENS, MAX_STATE_DIM, VLM_DIM, make_att_2d_masks,
                            resize_with_pad_uint8, sinusoidal_time_embedding)

# Engine name -> (ONNX file, its single input) for the one-input graphs.
GRAPHS = {
    "vision": ("smolvlm_vision.onnx", "image"),
    "prefill": ("smolvlm_expert_prefill.onnx", None),
    "decode": ("smolvlm_expert_decode.onnx", None),
    "state_proj": ("state_projector.onnx", "state"),
    "action_in": ("action_in_projector.onnx", "action"),
    "time_in": ("time_in_projector.onnx", "action_time"),
    "time_out": ("time_out_projector.onnx", "hidden"),
    "action_out": ("action_out_projector.onnx", "expert_out"),
}


class Bundle:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "export_info.json").read_text())
        if "embed_tokens" not in self.info:
            raise ValueError(f"{root} has no embed_tokens.npy: re-export it with "
                             "export/export.sh (bundles from Hugging Face do not carry it)")
        self.names = list(GRAPHS)
        self.n_cam_slots = int(self.info["n_cam_slots"])
        self.lang_len = int(self.info["lang_len"])
        self.prefix_len = int(self.info["prefix_len"])
        self.chunk_size = int(self.info["chunk_size"])
        self.num_steps = int(self.info["num_steps"])
        self.n_layers = int(self.info["vlm_layers"])
        self.embed = np.load(self.root / self.info["embed_tokens"], mmap_mode="r")
        self._tokenizer = None
        self._lang: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def onnx_path(self, name: str) -> Path:
        return self.root / GRAPHS[name][0]

    def tokens(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        """(ids [1,48], mask [1,48]) as the checkpoint's newline + tokenizer steps make them."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.root / "tokenizer"))
        task = task if task.endswith("\n") else task + "\n"
        tok = self._tokenizer(task, padding="max_length", padding_side="right",
                              max_length=self.lang_len, truncation=True, return_tensors="np")
        return tok["input_ids"].astype(np.int64), tok["attention_mask"].astype(bool)

    def language(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        """Prompt embedding rows (scaled as embed_language_tokens' caller does) and mask,
        cached per task."""
        if task not in self._lang:
            ids, mask = self.tokens(task)
            self._lang[task] = (self.embed_ids(ids), mask)
        return self._lang[task]

    def embed_ids(self, ids: np.ndarray) -> np.ndarray:
        emb = np.asarray(self.embed[ids[0]], np.float32)[None]
        return emb * np.float32(math.sqrt(emb.shape[-1]))


class Policy:
    """Engines + the constant empty-slot embedding."""

    def __init__(self, bundle: Bundle, run):
        self.b, self.run = bundle, run
        self.pad_cam = None
        if bundle.n_cam_slots > 1:
            self.pad_cam = self.vision(-np.ones((1, 3, 512, 512), np.float32))

    def one(self, name: str, x: np.ndarray) -> np.ndarray:
        return next(iter(self.run(name, {GRAPHS[name][1]: x}).values()))

    def vision(self, pixels: np.ndarray) -> np.ndarray:
        emb = self.one("vision", pixels)
        return emb * np.float32(math.sqrt(emb.shape[-1]))

    def sample(self, img_embs: list[np.ndarray], lang: tuple[np.ndarray, np.ndarray],
               state: np.ndarray, noise: np.ndarray, timings: dict | None = None
               ) -> np.ndarray:
        """x_t after num_steps: [1,chunk,32], normalized model space.

        img_embs: one [1,64,960] per real camera. state [1,32], normalized and padded.
        """
        b, t = self.b, ({} if timings is None else timings)
        t0 = time.perf_counter()
        lang_emb, lang_mask = lang
        n_real, n_pad = len(img_embs), b.n_cam_slots - len(img_embs)
        state_emb = self.one("state_proj", state).reshape(1, 1, VLM_DIM)
        embs = np.concatenate(img_embs + [self.pad_cam] * n_pad + [lang_emb, state_emb],
                              axis=1).astype(np.float32)
        pad_masks = np.concatenate(
            [np.ones((1, IMG_TOKENS), bool)] * n_real
            + [np.zeros((1, IMG_TOKENS), bool)] * n_pad
            + [lang_mask, np.ones((1, 1), bool)], axis=1)
        att_masks = np.zeros((1, b.prefix_len), bool)
        att_masks[0, -1] = True                      # the state token starts a new block
        kv = self.run("prefill", {
            "attention_mask": make_att_2d_masks(pad_masks, att_masks),
            "position_ids": (np.cumsum(pad_masks, axis=1) - 1).astype(np.int64),
            "vlm_embeds": embs})
        past = {k.replace("present_", "past_"): v for k, v in kv.items()}
        t1 = time.perf_counter()

        suffix = np.ones((1, b.chunk_size), bool)
        full_att = np.concatenate(
            [np.broadcast_to(pad_masks[:, None, :], (1, b.chunk_size, b.prefix_len)),
             make_att_2d_masks(suffix, suffix)], axis=2)
        pos = (pad_masks.sum(axis=-1, keepdims=True)
               + np.cumsum(suffix, axis=1) - 1).astype(np.int64)
        x_t = noise.astype(np.float32).copy()
        dt = -1.0 / b.num_steps
        for step in range(b.num_steps):
            tt = 1.0 + step * dt
            a = self.one("action_in", x_t)
            te = np.broadcast_to(sinusoidal_time_embedding(tt)[None, None, :], a.shape)
            h = self.one("time_in", np.concatenate([a, te], axis=2).astype(np.float32))
            h = h * (1.0 / (1.0 + np.exp(-h)))                                # SiLU
            suf = self.one("time_out", h)
            out = next(iter(self.run("decode", {"attention_mask": full_att,
                                                "position_ids": pos,
                                                "expert_embeds": suf, **past}).values()))
            x_t = x_t + np.float32(dt) * self.one("action_out", out)
        t["prefill"] = (t1 - t0) * 1e3
        t["denoise"] = (time.perf_counter() - t1) * 1e3
        return x_t


def preprocess(image_u8: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> [1,3,512,512] in [-1,1] (lerobot resize_with_pad, then SigLIP's
    [-1,1])."""
    return resize_with_pad_uint8(image_u8)


def pad_state(state: np.ndarray) -> np.ndarray:
    s = np.zeros((1, MAX_STATE_DIM), np.float32)
    flat = np.asarray(state, np.float32).ravel()
    s[0, :flat.shape[0]] = flat
    return s


def validate_fixture(bundle: Bundle, policy: Policy) -> dict:
    """Engines vs the stock-PyTorch FP32 chunk the exporter stored in the bundle.

    Same pixels, prompt, normalized state and noise as the reference; fails closed on the
    full padded chunk. `image_preprocess` and `input_ids_match` check this runtime's own
    host preprocessing and tokenizer against the stock pipeline's.
    """
    fx = bundle.info.get("fixture") or {}
    path = bundle.root / fx.get("file", "fixture.npz")
    if not path.exists():
        return {"status": "MISSING", "file": str(path)}
    r = np.load(path)
    pix = r["pixel_values"].astype(np.float32)
    embs = [policy.vision(pix[i:i + 1]) for i in range(len(pix))]
    lang = (bundle.embed_ids(r["lang_tokens"]), r["lang_masks"].astype(bool))
    act = policy.sample(embs, lang, r["model_state"].astype(np.float32), r["noise"])
    ref = r["action_pred"]
    pre = np.concatenate([preprocess(np.ascontiguousarray(im)) for im in r["raw"]])
    ids, mask = bundle.tokens(fx["task"])
    ids_match = bool(np.array_equal(ids, r["lang_tokens"])
                     and np.array_equal(mask, r["lang_masks"].astype(bool)))
    reports = {
        "action": _cmp(act[:, 0], ref[:, 0]),
        "chunk": _cmp(act, ref),
        "image_preprocess": _cmp(pre, pix),
    }
    ok = (reports["chunk"]["cosine"] >= FIXTURE_MIN_COSINE
          and reports["chunk"]["max_pct_range"] <= FIXTURE_MAX_PCT_RANGE and ids_match)
    return {"status": "PASS" if ok else "FAIL", "threshold": FIXTURE_MIN_COSINE,
            "max_pct_range_threshold": FIXTURE_MAX_PCT_RANGE, "input_ids_match": ids_match,
            "source": fx.get("source", "stock PyTorch float32"), "reports": reports}
