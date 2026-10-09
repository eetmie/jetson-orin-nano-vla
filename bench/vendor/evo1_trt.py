# ─────────────────────────────────────────────────────────────────────────────
# Written for this repo (not vendored). The host loop follows evo1_split_ort.py, which is
# the tested ORT implementation; the engines, serial builds and fixture comparison are
# GR00T's (groot_trt.py).
# ─────────────────────────────────────────────────────────────────────────────
"""EVO1 split-engine inference on the TensorRT runtime, numpy host glue.

The bundle's graphs, minus `token_embedding` (a single Gather): its table ships as
`embed_tokens.npy` and is memory-mapped, so only the prompt's rows are ever read.

Per observation (V views, `num_inference_timesteps` Euler steps):

    vision_k        [V,3,448,448] -> [V,256,896]
    host            scatter the image features over the prompt's image tokens
    language_k      InternVL3's Qwen2.5 layers + final norm -> fused tokens
    action_context  state token + per-block cross-attention K/V, once per observation
    steps x:        action_step (cached K/V) -> action_output -> velocity;
                    action += velocity / steps
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .groot_trt import FIXTURE_MAX_PCT_RANGE, FIXTURE_MIN_COSINE, _cmp

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"


def _chain(names: list[str], prefix: str) -> list[str]:
    out = [n for n in names if n.startswith(prefix + "_") and n.rsplit("_", 1)[1].isdigit()]
    return sorted(out, key=lambda n: int(n.rsplit("_", 1)[1]))


class Bundle:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.b = b = json.loads((self.root / "bundle.json").read_text())
        if b.get("model") != "evo1" or b.get("schema_version") != 1:
            raise ValueError(f"{root} is not an EVO1 split bundle")
        if "embed_tokens" not in b:
            raise ValueError(f"{root} has no embed_tokens.npy: re-export it with "
                             "export/export.sh")
        if bool(b.get("deployable")) == bool(b.get("random_action_head")):
            raise ValueError("bundle disagrees with itself about its action head")
        if b.get("max_views") != b.get("valid_views"):
            raise ValueError("runtime requires max_views == valid_views")
        graphs = [g["name"] for g in b["graphs"]]
        self.files = {g["name"]: self.root / g["file"] for g in b["graphs"]}
        self.vision = _chain(graphs, "vision")
        self.language = _chain(graphs, "language")
        self.names = self.vision + self.language + [
            "action_context", "action_step", "action_output"]
        self.views = int(b["valid_views"])
        self.hidden = int(b["hidden_size"])
        self.image_token = int(b["image_token_id"])
        self.steps = int(b["num_inference_timesteps"])
        self.embed = np.load(self.root / b["embed_tokens"], mmap_mode="r")
        self._tokenizer = None
        self._prompts: dict[str, tuple] = {}

    def onnx_path(self, name: str) -> Path:
        return self.files[name]

    def preprocess(self, images_u8: list[np.ndarray]) -> np.ndarray:
        """V x HxWx3 uint8 -> [V,3,448,448]: PIL bicubic resize, ImageNet normalization."""
        from PIL import Image

        size = int(self.b["image_size"])
        out = np.empty((len(images_u8), 3, size, size), np.float32)
        for i, view in enumerate(images_u8):
            r = Image.fromarray(np.ascontiguousarray(view)).resize(
                (size, size), Image.Resampling.BICUBIC)
            x = np.asarray(r, np.float32) / 255.0
            out[i] = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        return out

    def prompt(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        """(input_ids [1,S], context_mask [1,S]) as LeRobot's _build_multimodal_prompts
        and the embedder's tokenizer call make them; cached per task."""
        if task not in self._prompts:
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(self.root / self.b["tokenizer"]["path"]), local_files_only=True)
            per_view = int(self.b["image_seq_length"])
            text = "".join(
                f"Image-{i + 1}: {IMG_START_TOKEN + IMG_CONTEXT_TOKEN * per_view + IMG_END_TOKEN}\n"
                for i in range(self.views)) + task.strip()
            enc = self._tokenizer(text, return_tensors="np", padding="max_length",
                                  truncation=True, max_length=int(self.b["seq_len"]))
            ids = enc["input_ids"].astype(np.int64)
            if int((ids == self.image_token).sum()) != per_view * self.views:
                raise ValueError("the prompt was truncated into its image tokens")
            self._prompts[task] = (ids, enc["attention_mask"].astype(bool))
        return self._prompts[task]


def causal_mask(valid: np.ndarray) -> np.ndarray:
    """The additive causal + padding mask native LeRobot EVO1 uses."""
    n = valid.shape[1]
    allowed = np.tril(np.ones((n, n), bool))[None, None] & valid[:, None, None, :]
    return np.where(allowed, 0.0, -10000.0).astype(np.float32)


def infer(bundle: Bundle, run, pixel_values: np.ndarray, input_ids: np.ndarray,
          context_mask: np.ndarray, state: np.ndarray, noise: np.ndarray,
          timings: dict | None = None) -> dict:
    """{"vision", "fused", "action"} for one observation. state [1,max_state_dim]."""
    t = {} if timings is None else timings
    t0 = time.perf_counter()
    x = pixel_values
    for name in bundle.vision:
        x = next(iter(run(name, {"pixel_values" if name == bundle.vision[0]
                                 else "hidden_in": x}).values()))
    t1 = time.perf_counter()
    h = np.asarray(bundle.embed[input_ids[0]], np.float32)[None]
    h[0, input_ids[0] == bundle.image_token] = x.reshape(-1, bundle.hidden)
    mask = causal_mask(context_mask)
    for name in bundle.language:
        h = next(iter(run(name, {"hidden_in": h, "causal_mask": mask}).values()))
    t2 = time.perf_counter()
    cache = run("action_context", {"fused_tokens": h, "context_mask": context_mask,
                                   "state": state.astype(np.float32)})
    t3 = time.perf_counter()
    action = noise.astype(np.float32).copy()
    for i in range(bundle.steps):
        ti = np.asarray([min(int((i / bundle.steps) * 999), 999)], np.int64)
        hid = run("action_step", {"action": action, "time_index": ti, **cache})["action_hidden"]
        # A new array, never `+=`: Engines skips the upload of an input whose host array
        # is the same object as last time, so an in-place update would never reach the GPU.
        action = action + run("action_output", {"action_hidden": hid})["velocity"] / bundle.steps
    t4 = time.perf_counter()
    t.update(vision=(t1 - t0) * 1e3, language=(t2 - t1) * 1e3,
             action_context=(t3 - t2) * 1e3, denoise=(t4 - t3) * 1e3)
    return {"vision": x, "fused": h, "action": action}


def validate_fixture(bundle: Bundle, engines) -> dict:
    """Engines vs the native-LeRobot FP32 outputs the exporter stored in the bundle.

    Same pixels, prompt, state and noise as the reference; fails closed on the full
    chunk. Also reports the vision and fused-token boundaries, and checks this runtime's
    host preprocessing and prompt against the stored ones.
    """
    fx = bundle.b.get("fixture") or {}
    path = bundle.root / fx.get("file", "parity_fixture.npz")
    if not path.exists():
        return {"status": "MISSING", "file": str(path)}
    r = np.load(path)
    out = infer(bundle, engines.run, r["pixel_values"].astype(np.float32),
                r["input_ids"].astype(np.int64), r["context_mask"].astype(bool),
                r["state"], r["initial_noise"])
    ref = r["expected_action"]
    valid = np.broadcast_to(r["context_mask"][..., None], r["expected_fused"].shape)
    raw = r["raw_images"] if "raw_images" in r else r["raw_image"][None]
    ids, cmask = bundle.prompt(fx["task"])
    ids_match = bool(np.array_equal(ids, r["input_ids"])
                     and np.array_equal(cmask, r["context_mask"].astype(bool)))
    reports = {
        "action": _cmp(out["action"][:, 0], ref[:, 0]),
        "chunk": _cmp(out["action"], ref),
        "vision": _cmp(out["vision"], r["expected_vision"]),
        "fused_valid": _cmp(out["fused"][valid], r["expected_fused"][valid]),
        "image_preprocess": _cmp(bundle.preprocess(list(raw)), r["pixel_values"]),
    }
    ok = (reports["chunk"]["cosine"] >= FIXTURE_MIN_COSINE
          and reports["chunk"]["max_pct_range"] <= FIXTURE_MAX_PCT_RANGE and ids_match)
    return {"status": "PASS" if ok else "FAIL", "threshold": FIXTURE_MIN_COSINE,
            "max_pct_range_threshold": FIXTURE_MAX_PCT_RANGE, "input_ids_match": ids_match,
            "source": fx.get("reference", "native LeRobot EVO1 float32"),
            "reports": reports}
