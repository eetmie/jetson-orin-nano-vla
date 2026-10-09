# ─────────────────────────────────────────────────────────────────────────────
# Written for this repo (not vendored). The host loop follows xvla_split_ort.py, which
# is the tested ORT implementation; the engines, serial builds and fixture comparison
# are GR00T's (groot_trt.py).
# ─────────────────────────────────────────────────────────────────────────────
"""X-VLA split-engine inference on the TensorRT runtime, numpy host glue.

The same twelve graphs as the ORT backend, but each weight is held once: ORT's TensorRT
EP keeps the ONNX initializers in host memory next to the engine (~5.5 bytes/param
measured for X-VLA), and on unified memory both count.

Per observation (V real views of `num_image_views` slots, `steps` denoising steps):

    vision_k        [V,3,224,224] -> [V,T,1024]
    host            scatter the views into a zero [num_image_views,T,1024] buffer
    text_encoder_k  BART over (prompt tokens + view 0)
    cond            loop-invariant projections of the VLM features and the other views
    steps x:        x_t = x1 * t + action * (1 - t)   (not an Euler step)
                    denoise_0 .. denoise_3 -> the clean-action estimate
    host            sigmoid on the gripper channels
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .groot_trt import FIXTURE_MAX_PCT_RANGE, FIXTURE_MIN_COSINE, _cmp

# XVLAImageNetNormalizeProcessorStep, applied to [0,1] images.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# BaseActionSpace.postprocess applies a sigmoid to these channels after the loop; the
# pre-step zeroing is baked into denoise_0.
GRIPPER_BY_MODE = {"ee6d": (9, 19), "agibot_ee6d": (9, 19), "joint": (6, 13),
                   "so101_bimanual": (5, 11)}


def _chain(names: list[str], prefix: str) -> list[str]:
    """Graphs of one family in execution order: numeric, so denoise_10 follows denoise_9."""
    out = [n for n in names if n == prefix or n.startswith(prefix + "_")]
    return sorted(out, key=lambda n: int(n.rsplit("_", 1)[1]) if n != prefix else 0)


class Bundle:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.b = json.loads((self.root / "bundle.json").read_text())
        b = self.b
        if "denoise_0" not in {g["name"] for g in b["graphs"]} or "tokens_per_view" not in b:
            raise ValueError(f"{root} is not an X-VLA split bundle")
        if b.get("denoise_input_mode", "x_t") != "x_t":
            raise ValueError("this runtime forms x_t on the host; re-export without "
                             "--fuse-denoise-interpolation")
        self.names = [g["name"] for g in b["graphs"]]
        self.files = {g["name"]: self.root / g["file"] for g in b["graphs"]}
        self.vision = _chain(self.names, "vision")
        self.text = _chain(self.names, "text_encoder")
        self.denoise = _chain(self.names, "denoise")
        self.num_views = int(b["num_image_views"])
        self.valid_views = int(b["valid_views"])
        self.tokens_per_view = int(b["tokens_per_view"])
        self.lang_len = int(b["lang_len"])
        self.chunk_size = int(b["chunk_size"])
        self.state_dim = int(b["max_state_dim"])
        self.action_dim = int(b["max_action_dim"])
        self.steps = int(b["num_denoising_steps"])
        mode = b["action_mode"]
        if mode not in GRIPPER_BY_MODE:
            raise ValueError(f"action_mode {mode!r} has no gripper mapping here; "
                             "postprocess would be wrong")
        self.gripper = list(GRIPPER_BY_MODE[mode])
        self._tokenizer = None
        self._ids: dict[str, np.ndarray] = {}

    def onnx_path(self, name: str) -> Path:
        return self.files[name]

    def input_ids(self, task: str) -> np.ndarray:
        """Prompt ids, padded to lang_len like the checkpoint's tokenizer step. Cached
        per task: the prompt rarely changes between calls."""
        if task not in self._ids:
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(self.root / self.b["tokenizer"]["path"]), local_files_only=True)
            self._ids[task] = self._tokenizer(
                task, max_length=self.lang_len, padding="max_length", truncation=True,
                padding_side="right", return_tensors="np")["input_ids"].astype(np.int64)
        return self._ids[task]

    def preprocess(self, image_u8: np.ndarray, size: int = 224) -> np.ndarray:
        """uint8 HxWx3 -> [3,size,size]: /255, ImageNet normalize, then lerobot's
        resize_with_pad (bilinear, keep aspect, pad left and top with 0 in normalized
        space), the order XVLAPolicy applies them in."""
        import cv2

        x = image_u8.transpose(2, 0, 1).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        c, h, w = x.shape
        if (h, w) == (size, size):
            return x
        ratio = max(w / size, h / size)
        rh, rw = int(h / ratio), int(w / ratio)
        r = cv2.resize(x.transpose(1, 2, 0), (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, c), np.float32)
        canvas[size - rh:, size - rw:] = r
        return np.ascontiguousarray(canvas.transpose(2, 0, 1))


def encode(bundle: Bundle, run, pixel_values: np.ndarray, input_ids: np.ndarray,
           timings: dict | None = None) -> np.ndarray:
    """Cold path, once per observation: pixels [V,3,224,224] -> cond_tokens."""
    t = {} if timings is None else timings
    t0 = time.perf_counter()
    x = pixel_values
    for name in bundle.vision:
        x = next(iter(run(name, {"pixel_values" if name == bundle.vision[0]
                                 else "hidden_in": x}).values()))
    t1 = time.perf_counter()
    # forward_vlm scatters the real views into a zero buffer: a padded view reads as
    # exactly zero, not as the tower's response to a blank image.
    full = np.zeros((bundle.num_views, bundle.tokens_per_view, x.shape[-1]), np.float32)
    full[:len(x)] = x
    h = next(iter(run(bundle.text[0], {"input_ids": input_ids,
                                       "image_tokens": full[0:1]}).values()))
    for name in bundle.text[1:]:
        h = next(iter(run(name, {"hidden_in": h}).values()))
    t2 = time.perf_counter()
    cond = next(iter(run("cond", {"vlm_features": h,
                                  "aux_visual": full[1:].reshape(1, -1, full.shape[-1])}
                         ).values()))
    t3 = time.perf_counter()
    t.update(vision=(t1 - t0) * 1e3, text=(t2 - t1) * 1e3, cond=(t3 - t2) * 1e3)
    return cond


def infer(bundle: Bundle, run, pixel_values: np.ndarray, input_ids: np.ndarray,
          proprio: np.ndarray, x1: np.ndarray, timings: dict | None = None) -> np.ndarray:
    """One action chunk [1,chunk,action_dim]. proprio [1,state_dim], x1 [1,chunk,action_dim]."""
    t = {} if timings is None else timings
    cond = encode(bundle, run, pixel_values, input_ids, t)
    t0 = time.perf_counter()
    x1 = np.ascontiguousarray(x1, np.float32)
    action = np.zeros_like(x1)
    for i in range(bundle.steps, 0, -1):
        s = i / bundle.steps
        x_t = x1 * np.float32(s) + action * np.float32(1.0 - s)
        h = next(iter(run(bundle.denoise[0], {"x_t": x_t, "t": np.array([s], np.float32),
                                              "proprio": proprio,
                                              "cond_tokens": cond}).values()))
        for name in bundle.denoise[1:]:
            h = next(iter(run(name, {"hidden_in": h}).values()))
        action = h
    action = action.copy()
    g = bundle.gripper
    action[..., g] = 1.0 / (1.0 + np.exp(-action[..., g]))
    t["denoise"] = (time.perf_counter() - t0) * 1e3
    return action


def validate_fixture(bundle: Bundle, engines) -> dict:
    """Engines vs the stock-PyTorch FP32 chunk the exporter stored in the bundle.

    Same pixels, prompt ids, proprio and noise as the reference; fails closed on the full
    chunk. `image_preprocess` and `input_ids` check this runtime's own host preprocessing
    and tokenizer against the stock pipeline's for the same raw frames and task.
    """
    fx = bundle.b.get("fixture") or {}
    path = bundle.root / fx.get("file", "fixture.npz")
    if not path.exists():
        return {"status": "MISSING", "file": str(path)}
    r = np.load(path)
    act = infer(bundle, engines.run, r["pixel_values"].astype(np.float32),
                r["input_ids"].astype(np.int64), r["proprio"].astype(np.float32), r["x1"])
    ref = r["action_pred"]
    pre = np.stack([bundle.preprocess(np.ascontiguousarray(im)) for im in r["raw"]])
    ids_match = bool(np.array_equal(bundle.input_ids(fx["task"]), r["input_ids"]))
    reports = {
        "action": _cmp(act[:, 0], ref[:, 0]),
        "chunk": _cmp(act, ref),
        "image_preprocess": _cmp(pre, r["pixel_values"]),
    }
    ok = (reports["chunk"]["cosine"] >= FIXTURE_MIN_COSINE
          and reports["chunk"]["max_pct_range"] <= FIXTURE_MAX_PCT_RANGE and ids_match)
    return {"status": "PASS" if ok else "FAIL", "threshold": FIXTURE_MIN_COSINE,
            "max_pct_range_threshold": FIXTURE_MAX_PCT_RANGE, "input_ids_match": ids_match,
            "source": fx.get("source", "stock PyTorch float32"), "reports": reports}
