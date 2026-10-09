#!/usr/bin/env python3
"""Run stock X-VLA on seeded synthetic inputs and store the result in the bundle.

The inputs go through the checkpoint's own preprocessor pipeline (tokenizer, image to
float, ImageNet normalization) and the policy's own image padding, so the saved
`input_ids` / `pixel_values` / `proprio` are exactly what the model sees. The noise draw
`generate_actions` makes internally is replaced by a seeded one that is saved too, so the
board can feed its engines the same draw and compare the whole chunk.

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
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    bundle = json.loads((a.bundle / "bundle.json").read_text())
    task = a.task or (bundle.get("tasks") or ["pick up the cube and place it in the box"])[0]
    views = int(bundle["valid_views"])

    cfg = PreTrainedConfig.from_pretrained(str(a.checkpoint))
    cfg.device = "cpu"
    policy = XVLAPolicy.from_pretrained(str(a.checkpoint), config=cfg)
    policy.to("cpu").float().eval()
    pre, _ = make_pre_post_processors(
        cfg, pretrained_path=str(a.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}})

    rng = np.random.default_rng(a.seed)
    keys = list(cfg.image_features)[:views]
    raw = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in keys]
    state_dim = cfg.input_features["observation.state"].shape[0]
    state = rng.standard_normal(state_dim).astype(np.float32)
    obs = {k: torch.from_numpy(im.transpose(2, 0, 1).copy()).float() for k, im in zip(keys, raw)}
    obs["observation.state"] = torch.from_numpy(state)
    obs["task"] = task
    batch = pre(obs)

    model = policy.model
    x1 = torch.from_numpy(
        rng.standard_normal((1, model.chunk_size, model.dim_action)).astype(np.float32))
    inputs = policy._build_model_inputs(batch)
    real_randn = torch.randn
    torch.randn = lambda *s, **kw: x1.clone().to(kw.get("dtype") or x1.dtype)
    try:
        with torch.no_grad():
            action = model.generate_actions(**inputs, steps=cfg.num_denoising_steps)
    finally:
        torch.randn = real_randn

    np.savez(
        a.bundle / FIXTURE,
        raw=np.stack(raw),
        pixel_values=inputs["image_input"][0, :views].numpy(),
        input_ids=inputs["input_ids"].numpy(),
        proprio=inputs["proprio"].numpy(),
        domain_id=inputs["domain_id"].numpy(),
        state=state, x1=x1.numpy(), action_pred=action.numpy(),
        meta=np.array(json.dumps({"task": task, "views": views, "seed": a.seed,
                                  "steps": cfg.num_denoising_steps})),
    )
    bundle["fixture"] = {"file": FIXTURE, "task": task, "seed": a.seed,
                         "source": "stock LeRobot XVLAPolicy, float32 on the CPU"}
    (a.bundle / "bundle.json").write_text(json.dumps(bundle, indent=2))
    write_manifest(a.bundle)
    print(f"action_pred {tuple(action.shape)} -> {a.bundle / FIXTURE}")


if __name__ == "__main__":
    main()
