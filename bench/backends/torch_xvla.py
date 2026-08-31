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

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from ..vendor.xvla_bundle_contract import normalize_vector, unnormalize_vector
from .base import Backend, InferResult


def _processor_tokenizer_contract(checkpoint: Path) -> dict:
    """Read the tokenizer settings the checkpoint's policy preprocessor actually used.

    X-VLA's raw config and saved processor can disagree. In particular, the public
    base checkpoint records tokenizer_max_length=1024 in config.json while its policy
    preprocessor pads to 50 tokens. Using the raw config overflows the policy
    transformer's 512-position table and, more importantly, no longer matches an export
    traced with the processor contract.
    """
    path = Path(checkpoint) / "policy_preprocessor.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    for step in data.get("steps", []):
        if step.get("registry_name") == "tokenizer_processor":
            config = step.get("config") or {}
            max_length = config.get("max_length")
            if max_length is None or int(max_length) <= 0:
                raise ValueError(
                    f"{path} has an invalid tokenizer max_length: {max_length!r}")
            return {
                "source": str(path),
                "tokenizer_name": config.get("tokenizer_name"),
                "max_length": int(max_length),
                "padding": config.get("padding"),
                "padding_side": config.get("padding_side"),
                "truncation": config.get("truncation"),
            }
    raise ValueError(f"{path} has no tokenizer_processor step")


class TorchXVLABackend(Backend):
    name = "torch-xvla"
    noise_injected = True

    def __init__(self, checkpoint: Path, weights: str = "float32",
                 autocast: str = "off", device: str = "cuda",
                 tokenizer: str | None = None, num_steps: int | None = None,
                 valid_views: int = 1, domain_id: int = 0,
                 lang_len: int | None = None, expected_num_views: int | None = None,
                 processor_contract: dict | None = None,
                 expected_checkpoint_sha: str | None = None,
                 expected_tokenizer_sha: str | None = None,
                 compile_model: bool = False):
        self.checkpoint = Path(checkpoint)
        self.dtype_name = weights
        self.autocast_name = autocast
        self.device_name = device
        self.tokenizer_id = tokenizer
        self.num_steps = num_steps
        self.valid_views = valid_views
        self.domain_id = domain_id
        self.lang_len = lang_len
        self.expected_num_views = expected_num_views
        self.processor_contract = processor_contract
        self.expected_checkpoint_sha = expected_checkpoint_sha
        self.expected_tokenizer_sha = expected_tokenizer_sha
        self.compile_model = compile_model
        self.policy = None

    def load(self) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from transformers import AutoTokenizer
        from ..vendor.xvla_bundle_contract import tree_sha256
        from ..vendor.xvla_split_ort import preprocess_image

        self.torch = torch
        self.preprocess_image = preprocess_image
        self.device = torch.device(self.device_name)
        self.dtype = getattr(torch, self.dtype_name)
        if self.expected_checkpoint_sha is not None:
            actual_checkpoint_sha = tree_sha256(self.checkpoint)
            if actual_checkpoint_sha != self.expected_checkpoint_sha:
                raise ValueError("PyTorch checkpoint does not match split bundle identity")

        # The checkpoint's config records the device it was trained on and
        # from_pretrained honours it; set it explicitly so the benchmark decides.
        cfg = PreTrainedConfig.from_pretrained(str(self.checkpoint))
        cfg.device = self.device_name
        processor_contract = _processor_tokenizer_contract(self.checkpoint)
        processor_len = processor_contract.get("max_length")
        if self.lang_len is not None and processor_len is not None:
            # An explicit length is allowed for comparison with an export, but record
            # the disagreement rather than silently pretending it is the checkpoint's
            # native processor contract.
            self._processor_lang_len_mismatch = self.lang_len != processor_len
        else:
            self._processor_lang_len_mismatch = False
        self.lang_len_used = self.lang_len or processor_len
        if self.lang_len_used is None:
            raw_len = int(getattr(cfg, "tokenizer_max_length", 0) or 0)
            max_seq = int(getattr(cfg, "max_len_seq", 0) or 0)
            if raw_len <= 0 or (max_seq and raw_len > max_seq):
                raise ValueError(
                    "cannot recover X-VLA's tokenizer length from "
                    "policy_preprocessor.json; pass --lang-len from the exact export "
                    f"contract (raw config has {raw_len}, max_len_seq={max_seq})")
            self.lang_len_used = raw_len
        if self.lang_len_used <= 0:
            raise ValueError("lang_len must be positive")

        self.tokenizer_id = (self.tokenizer_id
                             or processor_contract.get("tokenizer_name")
                             or getattr(cfg, "tokenizer_name", None)
                             or "facebook/bart-large")
        self._tokenizer_contract_source = processor_contract.get("source")
        if self.expected_tokenizer_sha is not None:
            actual_tokenizer_sha = tree_sha256(Path(self.tokenizer_id))
            if actual_tokenizer_sha != self.expected_tokenizer_sha:
                raise ValueError("PyTorch tokenizer does not match split bundle identity")
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
        if self.expected_num_views is not None and self.num_views != self.expected_num_views:
            raise ValueError(
                f"checkpoint declares num_image_views={self.num_views}, but the split "
                f"bundle declares {self.expected_num_views}")
        if self.valid_views <= 0 or self.valid_views > self.num_views:
            raise ValueError(
                f"valid_views must be in [1, {self.num_views}], got {self.valid_views}")
        if self.processor_contract:
            if self.processor_contract.get("action_mode") != self.cfg.action_mode:
                raise ValueError("processor action mode does not match checkpoint config")
            self.physical_state_dim = int(self.processor_contract["state"]["dim"])
            self.physical_action_dim = int(self.processor_contract["action"]["dim"])
            if int(self.processor_contract["state"]["model_dim"]) != self.model.dim_proprio:
                raise ValueError("processor/model state dimensions disagree")
            if int(self.processor_contract["action"]["model_dim"]) != self.model.dim_action:
                raise ValueError("processor/model action dimensions disagree")
            if (self.cfg.action_mode == "auto"
                    and not self.processor_contract.get("physical_boundary_complete")):
                raise ValueError("action_mode='auto' requires complete processor stats")
            self._processor_contract_sha256 = hashlib.sha256(json.dumps(
                self.processor_contract, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
        else:
            self.physical_state_dim = self.model.dim_proprio
            self.physical_action_dim = None
            self._processor_contract_sha256 = None
        self.steps = (self.num_steps if self.num_steps is not None else int(
            getattr(self.cfg, "num_denoising_steps", 10)))
        if self.steps <= 0:
            raise ValueError("num_steps must be positive")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id, local_files_only=self.expected_tokenizer_sha is not None)
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
            "action_dim": self.physical_action_dim or int(self.model.dim_action),
            "max_action_dim": int(self.model.dim_action),
            "state_dim": self.physical_state_dim,
            "max_state_dim": int(self.model.dim_proprio),
            "num_views": self.num_views,
            "valid_views": self.valid_views,
            "resize": [224, 224],
            "domain_id": self.domain_id,
            "tokenizer": self.tokenizer_id,
            "lang_len": self.lang_len_used,
            "tokenizer_contract_source": self._tokenizer_contract_source,
            "processor_lang_len_mismatch": self._processor_lang_len_mismatch,
            "tokenizer_sha256": self.expected_tokenizer_sha,
            "stats_sha256": self._processor_contract_sha256,
            "checkpoint_tree_sha256": self.expected_checkpoint_sha,
            "physical_boundary_complete": bool(
                (self.processor_contract or {}).get("physical_boundary_complete")),
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
            tok = self.tokenizer(task, max_length=self.lang_len_used, padding="max_length",
                                 truncation=True, padding_side="right",
                                 return_tensors="pt")
            self._lang_cache[task] = tok["input_ids"].to(self.device)
        return self._lang_cache[task]

    def infer(self, obs: Observation) -> InferResult:
        torch = self.torch
        t0 = time.perf_counter()

        if len(obs.images) != self.valid_views:
            raise ValueError(
                f"observation has {len(obs.images)} views, expected {self.valid_views}")
        pixels = np.stack([self.preprocess_image(im)
                           for im in obs.images]).astype(np.float32)
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

        flat = np.asarray(obs.state, dtype=np.float32).ravel()
        if flat.size != self.physical_state_dim:
            raise ValueError(
                f"state must contain exactly {self.physical_state_dim} axes, got {flat.size}")
        if self.processor_contract:
            flat = normalize_vector(flat, self.processor_contract["state"])
        proprio = torch.zeros(1, self.model.dim_proprio, device=self.device,
                              dtype=self.dtype)
        proprio[0, :len(flat)] = torch.from_numpy(flat).to(self.device, self.dtype)
        domain_id = torch.tensor([self.domain_id], dtype=torch.long, device=self.device)
        t_h2d = time.perf_counter()

        target = (1, int(self.cfg.chunk_size), int(self.model.dim_action))
        noise = np.asarray(obs.noise, dtype=np.float32)
        if noise.shape != target:
            raise ValueError(f"noise must have shape {target}, got {noise.shape}")
        x1 = torch.from_numpy(noise).to(self.device, self.dtype)
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
        if self.physical_action_dim is not None:
            chunk = chunk[..., :self.physical_action_dim]
        self.last_normalized_action = chunk.copy()
        if self.processor_contract:
            chunk = unnormalize_vector(chunk, self.processor_contract["action"])
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
