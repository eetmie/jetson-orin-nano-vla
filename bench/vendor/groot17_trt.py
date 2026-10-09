# ─────────────────────────────────────────────────────────────────────────────
# VENDORED from spark-projects @ c73dea0 — vla-onnx/groot/pipeline17.py
#
# GR00T N1.7 host glue for the split bundle. Engines, serial builds and the fixture
# comparison are shared with N1.6 (groot_trt.py); only the bundle contract, the image
# preprocessing and the inference chain differ.
# ─────────────────────────────────────────────────────────────────────────────
"""GR00T N1.7 split-engine inference on the TensorRT runtime, numpy host glue.

Backbone is Cosmos-Reason2-2B (Qwen3-VL). Per observation (V views, 4 denoising steps):

    vision_k  [V,3,256,256] -> [V,64,2048] + three DeepStack features   per frame
    host      scatter image tokens into the prompt (right-padded to a fixed S) and the
              DeepStack features into zero [1,S,2048] buffers
    llm_k     16 decoder layers; llm_0/llm_1 add the DeepStack features; NO final norm
              (stock backbone_features is the pre-norm hidden_states[-1])
    cond_k    vlln + 4-layer VL self-attention (pads hidden) + state encoder
    4x:  time -> dit_0 .. dit_10   (last chunk applies the decoder and Euler step)

N1.7 sees each camera twice: the frame now and one `video_delta_indices[0]` frames
earlier. The ViT attends within one image, so a frame's vision output depends on that
frame alone; the runtime keeps the outputs of past frames and only encodes the new ones.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .groot_trt import FIXTURE_MAX_PCT_RANGE, FIXTURE_MIN_COSINE, _cmp, denoise


class Bundle:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.b = json.loads((self.root / "bundle.json").read_text())
        if not self.b.get("schema", "").startswith("groot-n1.7-split/"):
            raise ValueError(f"{root} is not a GR00T N1.7 split bundle")
        self.embed = np.load(self.root / self.b["embed_tokens"], mmap_mode="r")
        self.names = [g["name"] for g in self.b["graphs"]]
        self.graph = {g["name"]: g for g in self.b["graphs"]}
        self.vision = [n for n in self.names if n.startswith("vision_")]
        self.llm = [n for n in self.names if n.startswith("llm_")]
        self.cond = [n for n in self.names if n.startswith("cond_")]
        self.dit = [n for n in self.names if n.startswith("dit_")]
        self.mod = [n for n in self.names if n.startswith("mod_")]
        self.kv = [n for n in self.names if n.startswith("kv_")]
        self.consts = None
        ids = np.asarray(self.b["input_ids"], dtype=np.int64)
        n, neg = self.b["prompt_tokens"], np.float32(self.b["mask_neg"])
        valid = np.arange(ids.shape[0]) < n
        img = ids == self.b["image_token"]
        self.image_positions = img
        self.text_bias = np.where(~img & valid, 0.0, neg).astype(np.float32)[None, None]
        self.image_bias = np.where(img & valid, 0.0, neg).astype(np.float32)[None, None]
        self.pad_bias = np.where(valid, 0.0, neg).astype(np.float32)[None, None]
        # The prompt never changes within a bundle, so its embedded rows are gathered once.
        self.prompt_embeds = np.asarray(self.embed[ids], dtype=np.float32)[None]

    def preprocess(self, image_u8: np.ndarray) -> np.ndarray:
        """One HxWx3 uint8 frame -> [3,256,256] float32, as the stock eval pipeline does it.

        Gr00tN1d7Processor eval transform: letterbox to square, shortest edge to 256
        (INTER_AREA), center crop 0.95, shortest edge back to 256. Qwen3-VL's processor
        then has nothing to resize (256 is a multiple of 32) and applies (x/255-0.5)/0.5.
        """
        import cv2

        h, w = image_u8.shape[:2]
        if h != w:
            m = max(h, w)
            top, left = (m - h) // 2, (m - w) // 2
            image_u8 = cv2.copyMakeBorder(image_u8, top, m - h - top, left, m - w - left,
                                          cv2.BORDER_CONSTANT, value=0)
        edge = self.b.get("shortest_image_edge", 256)
        frac = self.b.get("crop_fraction", 0.95)

        def smallest_max(img):
            hh, ww = img.shape[:2]
            s = edge / min(hh, ww)
            if s == 1:
                return img
            return cv2.resize(img, (round(ww * s), round(hh * s)), interpolation=cv2.INTER_AREA)

        img = smallest_max(image_u8)
        hh, ww = img.shape[:2]
        ch, cw = int(hh * frac), int(ww * frac)
        y0, x0 = (hh - ch) // 2, (ww - cw) // 2
        img = smallest_max(img[y0:y0 + ch, x0:x0 + cw])
        if list(img.shape[:2]) != list(self.b["image_hw"]):
            raise ValueError(f"preprocessed frame is {img.shape[:2]}, the bundle expects "
                             f"{self.b['image_hw']}")
        x = img.astype(np.float32) / 255.0
        x = (x - self.b["image_mean"]) / self.b["image_std"]
        return np.ascontiguousarray(x.transpose(2, 0, 1))


def encode_frames(bundle: Bundle, run, pixel_values: np.ndarray) -> list[np.ndarray]:
    """One frame per view [V,3,H,W] -> [tokens, ds_0, ds_1, ds_2], each [V,64,2048]."""
    x, ds = pixel_values, {}
    for name in bundle.vision:
        o = run(name, {"pixel_values" if name == bundle.vision[0] else "x": x})
        x = o["vision_tokens" if name == bundle.vision[-1] else "x_out"]
        ds.update({k: v for k, v in o.items() if k.startswith("ds_")})
    return [x, *[ds[f"ds_{j}"] for j in range(bundle.b["deepstack"])]]


def infer(bundle: Bundle, run, frames: list[list[np.ndarray]], state: np.ndarray,
          noise: np.ndarray, timings: dict | None = None) -> np.ndarray:
    """One action chunk from already-encoded frames.

    frames: per time slot, oldest first, encode_frames() of that slot's V views.
    state [1,1,132] normalized, noise [1,40,132].
    """
    b = bundle.b
    t = {} if timings is None else timings
    t0 = time.perf_counter()
    sel = bundle.image_positions
    h = bundle.prompt_embeds.copy()
    h[0, sel] = np.concatenate([f[0].reshape(-1, f[0].shape[-1]) for f in frames])
    ds_full = []
    for j in range(b["deepstack"]):
        z = np.zeros_like(h)
        z[0, sel] = np.concatenate([f[1 + j].reshape(-1, f[1 + j].shape[-1]) for f in frames])
        ds_full.append(z)
    for name in bundle.llm:
        feeds = {"h": h}
        feeds.update({k: ds_full[int(k[3:])] for k in bundle.graph[name]["inputs"]
                      if k.startswith("ds_")})
        h = next(iter(run(name, feeds).values()))
    o = run(bundle.cond[0], {"features": h, "pad_bias": bundle.pad_bias,
                             "state": state.astype(np.float32)})
    vl, sf = o["vl"], o["state_features"]
    for name in bundle.cond[1:]:
        vl = run(name, {"vl": vl, "pad_bias": bundle.pad_bias})["vl_out"]
    t1 = time.perf_counter()
    actions = denoise(bundle, run, {"vl": vl, "state_features": sf,
                                    "text_bias": bundle.text_bias,
                                    "image_bias": bundle.image_bias}, noise)
    t2 = time.perf_counter()
    t.update(backbone=(t1 - t0) * 1e3, denoise=(t2 - t1) * 1e3)
    return actions


class FrameHistory:
    """Encoded frames by capture time, so the history slot reuses an earlier encode.

    `lag_s` is how far back the oldest slot looks (delta index / camera fps). The pick
    is the newest entry at least `lag_s` old, else the oldest one held; on the very
    first call that is the current frame itself.
    """

    def __init__(self, lag_s: float):
        self.lag_s = lag_s
        self.entries: list[tuple[float, list[np.ndarray]]] = []

    def push(self, t: float, enc: list[np.ndarray]) -> None:
        self.entries.append((t, enc))
        # Keep one entry older than the lag; everything before it can never be picked.
        old = [i for i, (ti, _) in enumerate(self.entries) if t - ti >= self.lag_s]
        if len(old) > 1:
            del self.entries[:old[-1]]

    def past(self, t: float) -> tuple[list[np.ndarray], float]:
        old = [(ti, e) for ti, e in self.entries if t - ti >= self.lag_s]
        ti, e = old[-1] if old else self.entries[0]
        return e, t - ti


def _unpatchify(pv: np.ndarray, n: int, gh: int, gw: int) -> np.ndarray:
    """Qwen3-VL processor patches [N*P, 3*2*16*16] -> [N,3,H,W] (temporal copy 0)."""
    x = pv.reshape(n, gh // 2, gw // 2, 2, 2, 3, 2, 16, 16)[..., 0, :, :]
    return np.ascontiguousarray(x.transpose(0, 5, 1, 3, 6, 2, 4, 7).reshape(n, 3, gh * 16, gw * 16))


def validate_fixture(bundle: Bundle, engines) -> dict:
    """Engines vs the stock-PyTorch FP32 outputs the exporter stored in the bundle.

    Same pixels (the stock processor's own), state and noise as the reference; fails
    closed on the full chunk. `image_preprocess` checks this runtime's host
    preprocessing against the stock processor's pixels for the same raw frames.
    """
    fx = bundle.b.get("fixture") or {}
    path = bundle.root / fx.get("file", "fixture.npz")
    if not path.exists():
        return {"status": "MISSING", "file": str(path)}
    r = np.load(path)
    thw = r["image_grid_thw"]
    pix = _unpatchify(r["pixel_values"].astype(np.float32), thw.shape[0], int(thw[0, 1]),
                      int(thw[0, 2]))
    v = bundle.b["views"]
    frames = [encode_frames(bundle, engines.run, pix[i * v:(i + 1) * v])
              for i in range(bundle.b["frames"])]
    act = infer(bundle, engines.run, frames, r["state"], r["noise"])
    ref = r["action_pred"]
    pre = np.stack([bundle.preprocess(np.ascontiguousarray(im)) for im in r["raw"]])
    used = bundle.b.get("action_used")
    reports = {
        "action": _cmp(act[:, 0], ref[:, 0]),
        "chunk": _cmp(act, ref),
        "image_preprocess": _cmp(pre, pix),
    }
    if used:
        reports["chunk_used_dims"] = _cmp(act[..., :used], ref[..., :used])
    ok = (reports["chunk"]["cosine"] >= FIXTURE_MIN_COSINE
          and reports["chunk"]["max_pct_range"] <= FIXTURE_MAX_PCT_RANGE)
    return {"status": "PASS" if ok else "FAIL", "threshold": FIXTURE_MIN_COSINE,
            "max_pct_range_threshold": FIXTURE_MAX_PCT_RANGE,
            "source": fx.get("source", "stock PyTorch float32"), "reports": reports}
