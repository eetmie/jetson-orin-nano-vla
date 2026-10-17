"""OpenPI loader and stock fixtures for the shared pi0.5 template exporter."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import sys

import numpy as np
import safetensors.torch as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi05"))
from bench.model_contract import validate
from bench.vendor.pi05_numeric import transform
from layers import array
from openpi_model import configure_downloads, train_config
from state_prompt import StatePrompt


def export(a, export_loaded):
    from openpi import transforms
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, create_sinusoidal_pos_embedding
    from openpi.policies.policy import Policy
    from openpi.shared import normalize

    spec = validate(json.loads(a.openpi_spec.read_text()))
    recipe = spec["openpi"]
    configure_downloads()
    train = train_config()
    cfg = dataclasses.replace(train.model, pytorch_compile_mode=None)
    if not getattr(cfg, "pi05", False):
        raise ValueError("the selected OpenPI training configuration is not pi05")
    if cfg.action_horizon != spec["chunk_size"] or cfg.action_dim != spec["max_action_dim"]:
        raise ValueError("OpenPI training configuration dimensions disagree with the inspected checkpoint")
    if cfg.action_dim != spec["max_state_dim"]:
        raise ValueError("OpenPI pads its state to action_dim; max_state_dim disagrees with the training configuration")
    if cfg.paligemma_variant != "gemma_2b" or cfg.action_expert_variant != "gemma_300m":
        raise ValueError("the pi05 template exporter supports gemma_2b with gemma_300m")
    discrete = bool(cfg.discrete_state_input)
    if "discrete_state_input" in recipe and discrete != recipe["discrete_state_input"]:
        raise ValueError("OpenPI training configuration disagrees with discrete_state_input in export_config.json")
    data = train.data.create(train.assets_dirs, cfg)
    if not data.asset_id:
        raise ValueError("OpenPI training configuration must identify checkpoint normalization assets")
    asset = a.checkpoint / "assets" / data.asset_id
    if not asset.resolve().is_relative_to((a.checkpoint / "assets").resolve()) or not (asset / "norm_stats.json").is_file():
        raise ValueError(f"checkpoint is missing normalization assets for {data.asset_id}")
    stats = normalize.load(asset)
    mode = "QUANTILES" if data.use_quantile_norm else "MEAN_STD"
    def numeric_stats(key, dim):
        if key not in stats:
            raise ValueError(f"checkpoint normalization assets have no {key} statistics")
        names = ("q01", "q99") if data.use_quantile_norm else ("mean", "std")
        result = {}
        for name in names:
            value = np.asarray(getattr(stats[key], name)).reshape(-1)
            if value.size < dim:
                raise ValueError(f"{key} normalization stats have {value.size} dimensions, expected {dim}")
            result[name] = value[:dim].tolist()
        return result
    state_stats = numeric_stats("state", spec["state_dim"])
    action_stats = numeric_stats("actions", spec["action_dim"])
    def normalize_state(x):
        return transform(x, state_stats, mode=mode, implementation="openpi", eps=1e-6)
    def postprocess(x):
        return transform(x[..., :spec["action_dim"]], action_stats, mode=mode,
                         implementation="openpi", eps=1e-6, inverse=True)
    if train.name != "pi05_libero" and recipe.get("output_transform") != "slice":
        raise ValueError("custom OpenPI export_config.json must declare output_transform='slice'; other robot output transforms need a runtime implementation")

    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixtures").mkdir()
    model = PI0Pytorch(cfg)
    missing, unexpected = st.load_model(model, str(a.checkpoint / "model.safetensors"), strict=False)
    if unexpected or any("embed_tokens" not in key for key in missing):
        raise ValueError(f"checkpoint does not match the selected model: missing={missing}, unexpected={unexpected}")
    # Match the LeRobot path's reference precision. BF16 arithmetic itself can exceed
    # the stock-policy parity gate, especially for a lightly fine-tuned action head.
    model.float().eval().cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    policy = Policy(model,
        transforms=[transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs,
                    transforms.Normalize(stats, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs],
        output_transforms=[*data.model_transforms.outputs,
                           transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm), *data.data_transforms.outputs],
        sample_kwargs={"num_steps": spec["num_steps"]}, pytorch_device="cuda", is_pytorch=True)
    captured = {}
    original = policy._sample_actions
    def sample(device, observation, **kwargs):
        result = original(device, observation, **kwargs)
        captured.update(observation=observation, actions=array(result).astype(np.float32))
        return result
    policy._sample_actions = sample
    prompt = StatePrompt(os.environ["PI05_TOKENIZER"]) if discrete else None
    tasks, table, fixtures = spec["tasks"], {}, []
    checked = sum(prompt.verify(t, spec["state_dim"]) for t in tasks) if discrete else 0
    rng = np.random.default_rng(a.seed)
    # Every task gets a fixture; the last also tests out-of-quantile state values.
    for i in range(max(a.fixtures, len(tasks))):
        task = tasks[i % len(tasks)]
        raw = rng.integers(0, 256, (spec["views"], 480, 640, 3), dtype=np.uint8)
        normalized = rng.uniform(-1.5 if i == max(a.fixtures, len(tasks)) - 1 else -1,
                                 1.5 if i == max(a.fixtures, len(tasks)) - 1 else 1, spec["state_dim"]).astype(np.float32)
        state = transform(normalized, state_stats, mode=mode, implementation="openpi", eps=1e-6, inverse=True).astype(np.float32)
        noise = rng.standard_normal((1, spec["chunk_size"], spec["max_action_dim"])).astype(np.float32)
        obs = {key: image for key, image in zip(recipe["raw_image_keys"], raw)}
        obs.update({recipe["state_key"]: state.copy(), "prompt": task})
        with torch.inference_mode():
            result = policy.infer(obs, noise=noise)
        observation = captured["observation"]
        keys = [key for key in observation.images if bool(observation.image_masks[key].item())]
        if keys != spec["image_keys"]:
            raise ValueError(f"stock OpenPI camera order {keys} disagrees with inspected {spec['image_keys']}")
        pixels = []
        for key in keys:
            pixel = array(observation.images[key]).astype(np.float32)
            if pixel.shape[-1] == 3:
                pixel = pixel.transpose(0, 3, 1, 2)
            if pixel.shape != (1, 3, 224, 224):
                raise ValueError(f"OpenPI model transforms produced image shape {pixel.shape}; pi05 templates require [1,3,224,224]")
            pixels.append(pixel[0])
        mask = array(observation.tokenized_prompt_mask).astype(bool)[0]
        tokens = array(observation.tokenized_prompt)[0][mask].astype(np.int64)
        state_norm = normalize_state(state)
        if observation.state.shape[-1] != spec["max_state_dim"]:
            raise ValueError("stock OpenPI state padding disagrees with the inspected checkpoint")
        if not np.allclose(state_norm, array(observation.state)[0, :spec["state_dim"]], rtol=0, atol=1e-5):
            raise ValueError("OpenPI input transforms change the state; this transform needs a runtime implementation")
        if discrete:
            bins = np.digitize(state_norm, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            if prompt.compose(prompt.head(task), bins) != tokens.tolist():
                raise ValueError("stock OpenPI tokenization disagrees with the runtime state prompt")
            table[task] = {"head": prompt.head(task)}
        else:
            table[task] = {"tokens": tokens.tolist(), "mask": [True] * len(tokens)}
        stock = captured["actions"]
        robot = np.asarray(result["actions"], np.float32)
        if stock.shape != noise.shape or robot.shape != (spec["chunk_size"], spec["action_dim"]):
            raise ValueError("stock OpenPI action shapes disagree with the inspected checkpoint")
        if not np.allclose(postprocess(stock[0]), robot, rtol=0, atol=1e-5):
            raise ValueError("OpenPI output transforms differ from normalization + slicing; this transform needs a runtime implementation")
        fixtures.append({"task": task, "raw": raw, "pixels": np.stack(pixels), "state": state,
                         "state_norm": state_norm, "tokens": tokens, "noise": noise,
                         "stock_actions": stock, "stock_robot": robot})
        print(f"OpenPI stock fixture {i}: {task!r}, state {len(state)}, {len(keys)} cameras", flush=True)
    policy._sample_actions = original
    del original, policy, captured, observation
    text_capacity = (max(len(prompt.head(t)) for t in tasks) + 4 * spec["state_dim"] + len(prompt.tail)
                     if discrete else max(len(entry["tokens"]) for entry in table.values()))
    if text_capacity > cfg.max_token_len:
        raise ValueError(f"prompt needs {text_capacity} tokens but OpenPI config allows {cfg.max_token_len}")
    capacity = spec["views"] * 256 + text_capacity
    if capacity + spec["chunk_size"] >= 2048:
        raise ValueError("camera/prompt/action sequence exceeds this exporter's 2048-position RoPE table")
    metadata = {"policy": "pi05_openpi", "policy_config": train.name, "schedule": "openpi", "stock_precision": "float32",
                "prompt_mode": "state" if discrete else "static",
                "image_keys": spec["image_keys"], "valid_cameras": spec["views"],
                "state_dim": spec["state_dim"], "action_horizon": spec["chunk_size"],
                "action_dim": spec["max_action_dim"], "robot_action_dim": spec["action_dim"],
                "num_steps": spec["num_steps"], "prefix_tokens": capacity, "prompt_capacity": text_capacity,
                "normalization": {"mode": mode, "implementation": "openpi", "state": state_stats,
                                  "state_eps": 1e-6, "action": action_stats, "action_eps": 1e-6},
                "tokenizer": ({"name": "openpi/PaliGemma", "max_length": cfg.max_token_len,
                               "sep": prompt.sep, "neg": prompt.neg, "pieces": prompt.pieces,
                               "tail": prompt.tail, "assembly_checked_prompts": checked} if discrete else {}),
                "tasks": table, "image_preprocess": "OpenPI: antialiased resize_with_pad to 224, uint8/255*2-1"}
    export_loaded(a, model, fixtures, metadata, create_sinusoidal_pos_embedding, postprocess)
