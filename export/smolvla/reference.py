#!/usr/bin/env python3
"""Run stock SmolVLA on seeded synthetic inputs and store the result in the bundle.

Also writes the token-embedding table as `embed_tokens.npy`, which the TensorRT runtime
memory-maps instead of running `smolvlm_text.onnx` (a single Gather).

The inputs go through the checkpoint's own preprocessor pipeline (newline, tokenizer,
state normalization) and the policy's own image resize/pad, so the saved tensors are
exactly what the model sees. The noise is drawn here and passed to `sample_actions`, so
the board can feed its engines the same draw and compare the whole padded chunk.

Runs on the CPU in float32, the precision the graphs were traced in.

    python reference.py --checkpoint <dir> --bundle <bundle dir> [--task "..."]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_common.bundle import write_manifest  # noqa: E402

FIXTURE = "fixture.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--task", default=None, help="default: the bundle's first task")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    info_path = a.bundle / "export_info.json"
    info = json.loads(info_path.read_text())
    task = a.task or (info.get("tasks") or ["pick up the cube and place it in the box"])[0]
    views = int(info["n_cam_slots"])

    cfg = PreTrainedConfig.from_pretrained(str(a.checkpoint))
    cfg.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(str(a.checkpoint), config=cfg)
    policy.to("cpu").float().eval()
    pre, _ = make_pre_post_processors(
        cfg, pretrained_path=str(a.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}})

    rng = np.random.default_rng(a.seed)
    keys = list(cfg.image_features)[:views]
    raw = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in keys]
    state_dim = cfg.input_features["observation.state"].shape[0]
    state = rng.standard_normal(state_dim).astype(np.float32)
    obs = {k: torch.from_numpy(im.transpose(2, 0, 1).copy()).float() / 255.0
           for k, im in zip(keys, raw)}
    obs["observation.state"] = torch.from_numpy(state)
    obs["task"] = task
    batch = pre(obs)

    images, img_masks = policy.prepare_images(batch)
    model_state = policy.prepare_state(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]
    noise = torch.from_numpy(rng.standard_normal(
        (1, cfg.chunk_size, cfg.max_action_dim)).astype(np.float32))
    with torch.no_grad():
        action = policy.model.sample_actions(images, img_masks, tokens, masks, model_state,
                                             noise=noise)

    table = policy.model.vlm_with_expert.vlm.model.text_model.get_input_embeddings().weight
    np.save(a.bundle / "embed_tokens.npy", table.detach().numpy().astype(np.float32))

    np.savez(
        a.bundle / FIXTURE,
        raw=np.stack(raw),
        pixel_values=torch.cat(images).numpy(),
        img_masks=torch.cat(img_masks).numpy(),
        lang_tokens=tokens.numpy(), lang_masks=masks.numpy(),
        state=state, model_state=model_state.numpy(),
        noise=noise.numpy(), action_pred=action.numpy(),
        meta=np.array(json.dumps({"task": task, "views": views, "seed": a.seed,
                                  "steps": cfg.num_steps})),
    )
    info["embed_tokens"] = "embed_tokens.npy"
    info["fixture"] = {"file": FIXTURE, "task": task, "seed": a.seed,
                       "source": "stock LeRobot SmolVLAPolicy, float32 on the CPU"}
    info_path.write_text(json.dumps(info, indent=2))
    write_manifest(a.bundle)
    print(f"action_pred {tuple(action.shape)} -> {a.bundle / FIXTURE}")


if __name__ == "__main__":
    main()
