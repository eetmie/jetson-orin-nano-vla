"""Backend: stock PyTorch — LeRobot's `SmolVLAPolicy`, straight off the checkpoint.

The control baseline. No ONNX, no TensorRT, no export step: load the fine-tuned
checkpoint the way the training repo does and call the model. Everything the other
backends buy has to be paid for against this number.

Entry point
-----------
`policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state,
noise=...)` — the same call the Spark-side parity script uses against the split
export, which is what makes the two directly comparable. It takes an already-resized
image and an already-normalized state and returns a normalized chunk, so the
harness's shared preprocessing and MEAN_STD stats apply unchanged. Going through
`select_action` instead would fold LeRobot's preprocessor pipeline and an action
queue into the measurement — useful to know, but a different question.

Precision — and why there are two FP16 runs, not one
----------------------------------------------------
The Orin Nano is compute 8.7: `platform_has_fast_fp16 = True`,
`platform_has_fast_bf16 = n/a`. FP16 is the only fast reduced precision the board
has, so "the PyTorch baseline" has to mean an FP16 PyTorch baseline. There are two
different things that can mean, and they are not interchangeable:

  `--weights float32 --autocast float16`
      Mixed precision the way LeRobot itself runs it (`use_amp: true` in the
      checkpoint's own train_config). Matmuls run in FP16, master weights stay FP32.
      This is *the* default-PyTorch path and the honest baseline.

  `--weights float16`
      A hard cast of the weights. Halves resident memory, which on an 8 GB board is
      the whole argument for doing it — but **LeRobot 0.5.1 does not support it**:
      `modeling_smolvla.py` hardcodes `suffix_out.to(dtype=torch.float32)` inside the
      denoise step, so `action_out_proj` gets an FP32 activation against FP16 weights
      and the call dies with "mat1 and mat2 must have the same dtype". Pass
      `--patch-half-out` to wrap that one projection so it casts its input to its own
      dtype. It is a deliberate, single-line deviation from stock LeRobot, recorded in
      `meta()` as `patched_half_out: true` — a fair benchmark can measure a patched
      path as long as it says so.

Either way, watch parity. SmolVLA trains in BF16, and putting the *whole* graph in
FP16 is precisely what overflowed the SigLIP vision tower on Blackwell (cosine 0.805,
730 constants past FP16's exponent range). The TensorRT path avoids that by keeping
norms in FP32 and letting rejected ops fall back; a blanket cast does not. If FP16
torch collapses here, that is a result — `bench parity` is what catches it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from ..vendor.smolvla_split import LANG_LEN, MAX_STATE_DIM, resize_with_pad_uint8
from .base import Backend, InferResult, load_export_info, load_stats

_DTYPES = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}


class TorchSmolVLABackend(Backend):
    name = "torch-smolvla"
    noise_injected = True

    def __init__(self, checkpoint: Path, bundle: Path | None = None,
                 weights: str = "float32", autocast: str = "off",
                 device: str = "cuda", action_dim: int = 4,
                 tokenizer_dir: Path | None = None, compile_model: bool = False,
                 patch_half_out: bool = False, cam_slots: int | None = None):
        self.checkpoint = Path(checkpoint)
        self.bundle = Path(bundle) if bundle else None
        self.dtype_name = _DTYPES[weights]
        self.autocast_name = autocast
        self.patch_half_out = patch_half_out
        self.device_name = device
        self.action_dim = action_dim
        self.tokenizer_dir = tokenizer_dir
        self.compile_model = compile_model
        # Camera SLOTS, which is not the same as cameras. An ONNX export bakes its slot
        # count into the prefix; PyTorch builds a prefix from however many images it is
        # handed. Feeding torch one image against a two-slot export means a 113-token
        # prefix against a 177-token one — a structural difference that would surface as
        # a parity failure and read like a numerics bug. Padding to the export's slot
        # count keeps the two comparable.
        self.cam_slots = cam_slots
        self.policy = None
        self._info = load_export_info(self.bundle) if self.bundle else {}

    def load(self) -> None:
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = torch.device(self.device_name)
        self.dtype = getattr(torch, self.dtype_name)

        policy = SmolVLAPolicy.from_pretrained(str(self.checkpoint)).eval()
        policy = policy.to(device=self.device, dtype=self.dtype)
        for p in policy.parameters():
            p.requires_grad_(False)
        if self.patch_half_out and self.dtype != torch.float32:
            policy.model.action_out_proj = _cast_to_weight_dtype(
                policy.model.action_out_proj)
        if self.compile_model:
            policy.model = torch.compile(policy.model)
        self.policy = policy
        self.cfg = policy.config

        # Same tokenizer as the split bundle when one is available, so token ids are
        # identical and a parity difference can only come from the runtime.
        tok_dir = self.tokenizer_dir or (self.bundle / "tokenizer" if self.bundle else None)
        self.tokenizer = (AutoTokenizer.from_pretrained(str(tok_dir)) if tok_dir
                          and Path(tok_dir).exists()
                          else policy.language_tokenizer)
        self.norm = load_stats(self.bundle) if self.bundle else load_stats(Path("."))
        self._lang_cache: dict[str, tuple] = {}

    def meta(self) -> dict:
        import torch
        m = {
            "backend": self.name,
            "family": "smolvla",
            "checkpoint": str(self.checkpoint),
            "weights_dtype": self.dtype_name,
            "autocast": self.autocast_name,
            "patched_half_out": bool(self.patch_half_out
                                     and self.dtype_name != "float32"),
            "device": self.device_name,
            "chunk_size": int(self.cfg.chunk_size),
            "num_steps": int(self.cfg.num_steps),
            "max_action_dim": int(self.cfg.max_action_dim),
            "action_dim": self.action_dim,
            "resize": list(self.cfg.resize_imgs_with_padding),
            "cam_slots": self.cam_slots,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "compiled": self.compile_model,
            "export_info": self._info,
        }
        if torch.cuda.is_available():
            m["gpu_name"] = torch.cuda.get_device_name(0)
            m["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        return m

    def _language(self, task: str):
        if task not in self._lang_cache:
            text = task if task.endswith("\n") else task + "\n"
            tok = self.tokenizer(text, padding="max_length", padding_side="right",
                                 max_length=LANG_LEN, truncation=True,
                                 return_tensors="np")
            ids = self.torch.from_numpy(tok["input_ids"].astype(np.int64)).to(self.device)
            mask = self.torch.from_numpy(
                tok["attention_mask"].astype(bool)).to(self.device)
            self._lang_cache[task] = (ids, mask)
        return self._lang_cache[task]

    def infer(self, obs: Observation) -> InferResult:
        torch = self.torch
        t0 = time.perf_counter()

        # -- preprocess: identical maths to the split path, on the CPU in numpy --
        imgs = [resize_with_pad_uint8(im) for im in obs.images]   # [1,3,512,512] in [-1,1]
        s = self.norm.normalize_state(np.asarray(obs.state, dtype=np.float32).reshape(-1))
        s_pad = np.zeros((1, MAX_STATE_DIM), dtype=np.float32)
        s_pad[0, :s.shape[0]] = s
        lang_ids, lang_mask = self._language(obs.task)
        t_pre = time.perf_counter()

        image_ts = [torch.from_numpy(i).to(self.device, self.dtype) for i in imgs]
        img_masks = [torch.ones(1, dtype=torch.bool, device=self.device)
                     for _ in image_ts]
        # lerobot's padding convention: an all -1 image (i.e. 0 before the SigLIP
        # [-1,1] normalization) behind a False mask. Same thing the split runtime
        # caches as its empty-slot embedding.
        n_pad = max(0, (self.cam_slots or len(image_ts)) - len(image_ts))
        for _ in range(n_pad):
            image_ts.append(torch.full_like(image_ts[0], -1.0))
            img_masks.append(torch.zeros(1, dtype=torch.bool, device=self.device))
        state_t = torch.from_numpy(s_pad).to(self.device, self.dtype)
        noise_t = torch.from_numpy(obs.noise).to(self.device, self.dtype)
        t_h2d = time.perf_counter()

        with torch.no_grad(), self._autocast():
            out = self.policy.model.sample_actions(
                image_ts, img_masks, lang_ids, lang_mask, state_t, noise=noise_t)
        if self.device.type == "cuda":
            torch.cuda.synchronize()          # the kernels are async; time the work
        t_model = time.perf_counter()

        actions = out.float().cpu().numpy()[0, :, :self.action_dim]
        chunk = self.norm.unnormalize_action(actions)
        t1 = time.perf_counter()

        return InferResult(chunk, {
            "total": (t1 - t0) * 1000.0,
            "preprocess": (t_pre - t0) * 1000.0,
            "host_to_device": (t_h2d - t_pre) * 1000.0,
            "model": (t_model - t_h2d) * 1000.0,
            "postprocess": (t1 - t_model) * 1000.0,
        })

    def _autocast(self):
        import contextlib
        if self.autocast_name == "off":
            return contextlib.nullcontext()
        return self.torch.autocast(device_type=self.device.type,
                                   dtype=getattr(self.torch, self.autocast_name))

    def close(self) -> None:
        try:
            import torch
            self.policy = None
            torch.cuda.empty_cache()
        except Exception:
            pass


def _cast_to_weight_dtype(inner):
    """Wrap a Linear so it casts its input to its own weight dtype.

    Exists only to work around LeRobot 0.5.1's hardcoded
    `suffix_out.to(dtype=torch.float32)` in `denoise_step`, which makes a
    half-weights policy unrunnable. Used only under `--patch-half-out`, and always
    disclosed in the run's metadata. Built here rather than at import time so this
    module still imports on a machine with no torch.
    """
    import torch.nn as nn

    class _CastToWeightDtype(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x.to(self.inner.weight.dtype))

    return _CastToWeightDtype(inner)
