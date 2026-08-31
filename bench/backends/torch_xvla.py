"""Backend: stock PyTorch — LeRobot `XVLAPolicy`, straight off the checkpoint.

The X-VLA half of the control baseline. Same role as `torch_smolvla`: load the
checkpoint the way the training repo does and call the model, so everything the
exported runtimes buy is measured against it.

Entry point is `model.generate_actions(input_ids, image_input, image_mask, domain_id,
proprio, steps)`, matching the reference emitter the split export is parity-checked
with.

Injecting the noise takes a detour. SmolVLA's `sample_actions` accepts `noise=`;
X-VLA's `generate_actions` draws its own `x1` internally with `torch.randn`. To feed
the same draw every backend gets, `torch.randn` is swapped for the duration of the call
and only intercepted for the exact target shape — everything else falls through to the
real one. Ugly, deliberate, and confined to a `try/finally`; without it there is no
element-wise comparison against the split export at all, only a distribution one.

Preprocessing (resize-with-pad to 224, ImageNet normalization) is taken from the
vendored X-VLA runtime so both backends see byte-identical pixels, and padded views are
zero-filled exactly as `forward_vlm` expects.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from ..vendor.xvla_split_ort import preprocess_image
from .base import Backend, InferResult


class TorchXVLABackend(Backend):
    name = "torch-xvla"
    noise_injected = True

    def __init__(self, checkpoint: Path, weights: str = "float32",
                 autocast: str = "off", device: str = "cuda",
                 tokenizer: str = "facebook/bart-large", num_steps: int | None = None,
                 valid_views: int = 1, domain_id: int = 0,
                 lang_len: int | None = None, compile_model: bool = False):
        self.checkpoint = Path(checkpoint)
        self.dtype_name = weights
        self.autocast_name = autocast
        self.device_name = device
        self.tokenizer_id = tokenizer
        self.num_steps = num_steps
        self.valid_views = valid_views
        self.domain_id = domain_id
        self.lang_len = lang_len
        self.compile_model = compile_model
        self.policy = None

    def load(self) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = torch.device(self.device_name)
        self.dtype = getattr(torch, self.dtype_name)

        # The checkpoint's config records the device it was trained on and
        # from_pretrained honours it; set it explicitly so the benchmark decides.
        cfg = PreTrainedConfig.from_pretrained(str(self.checkpoint))
        cfg.device = self.device_name
        policy = XVLAPolicy.from_pretrained(str(self.checkpoint), config=cfg).eval()
        policy = policy.to(device=self.device, dtype=self.dtype)
        for p in policy.parameters():
            p.requires_grad_(False)
        if self.compile_model:
            policy.model = torch.compile(policy.model)

        self.policy = policy
        self.cfg = policy.config
        self.model = policy.model
        self.num_views = int(self.cfg.num_image_views)
        self.steps = (self.num_steps if self.num_steps is not None else int(
            getattr(self.cfg, "num_denoising_steps", 10)))
        if self.steps <= 0:
            raise ValueError("num_steps must be positive")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
        self._lang_cache: dict[str, object] = {}

    def artifact_paths(self) -> dict[str, Path]:
        return {"checkpoint": self.checkpoint}

    def meta(self) -> dict:
        import torch
        m = {
            "backend": self.name,
            "family": "xvla",
            "checkpoint": str(self.checkpoint),
            "weights_dtype": self.dtype_name,
            "autocast": self.autocast_name,
            "device": self.device_name,
            "chunk_size": int(self.cfg.chunk_size),
            "num_steps": self.steps,
            "action_dim": int(self.model.dim_action),
            "state_dim": int(self.model.dim_proprio),
            "num_views": self.num_views,
            "valid_views": self.valid_views,
            "resize": [224, 224],
            "domain_id": self.domain_id,
            "tokenizer": self.tokenizer_id,
            "torch": torch.__version__,
            "compiled": self.compile_model,
            "kv_cache": False,
        }
        if torch.cuda.is_available():
            m["gpu_name"] = torch.cuda.get_device_name(0)
            m["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        return m

    def _language(self, task: str):
        if task not in self._lang_cache:
            n = self.lang_len or int(getattr(self.cfg, "tokenizer_max_length", 32))
            tok = self.tokenizer(task, max_length=n, padding="max_length",
                                 truncation=True, padding_side="right",
                                 return_tensors="pt")
            self._lang_cache[task] = tok["input_ids"].to(self.device)
        return self._lang_cache[task]

    def infer(self, obs: Observation) -> InferResult:
        torch = self.torch
        t0 = time.perf_counter()

        pixels = np.stack([preprocess_image(im)
                           for im in obs.images[:self.valid_views]]).astype(np.float32)
        input_ids = self._language(obs.task)
        t_pre = time.perf_counter()

        image_input = torch.from_numpy(pixels).unsqueeze(0).to(self.device, self.dtype)
        n_pad = self.num_views - image_input.shape[1]
        if n_pad > 0:
            # forward_vlm scatters valid views into a zero buffer; padded views must be
            # zeros and are never given a forward pass.
            image_input = torch.cat(
                [image_input,
                 image_input.new_zeros((1, n_pad, *image_input.shape[2:]))], dim=1)
        image_mask = torch.zeros(1, self.num_views, dtype=torch.bool,
                                 device=self.device)
        image_mask[0, :self.valid_views] = True

        proprio = torch.zeros(1, self.model.dim_proprio, device=self.device,
                              dtype=self.dtype)
        flat = np.asarray(obs.state, dtype=np.float32).ravel()[:self.model.dim_proprio]
        proprio[0, :len(flat)] = torch.from_numpy(flat).to(self.device, self.dtype)
        domain_id = torch.tensor([self.domain_id], dtype=torch.long, device=self.device)
        t_h2d = time.perf_counter()

        target = (1, int(self.cfg.chunk_size), int(self.model.dim_action))
        x1 = torch.from_numpy(
            obs.noise.astype(np.float32)[:, :target[1], :target[2]]).to(self.device,
                                                                       self.dtype)
        real_randn = torch.randn

        def fixed_randn(*a, **kw):
            shape = tuple(a[0]) if len(a) == 1 and not isinstance(a[0], int) else tuple(a)
            return x1 if shape == target else real_randn(*a, **kw)

        torch.randn = fixed_randn
        try:
            with torch.no_grad(), self._autocast():
                action = self.model.generate_actions(
                    input_ids=input_ids, image_input=image_input,
                    image_mask=image_mask, domain_id=domain_id, proprio=proprio,
                    steps=self.steps)
        finally:
            torch.randn = real_randn
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_model = time.perf_counter()

        chunk = action.float().cpu().numpy()
        if chunk.ndim == 3:
            chunk = chunk[0]
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
            self.policy = self.model = None
            torch.cuda.empty_cache()
        except Exception:
            pass
