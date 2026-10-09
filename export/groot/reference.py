#!/usr/bin/env python3
# VENDORED from spark-projects @ 9e367cd — vla-onnx/groot/reference.py
"""Run stock GR00T N1.6 on seeded synthetic inputs and save everything parity needs.

The inputs go through the checkpoint's own eval image transform, Eagle chat template and
tokenizer, so the saved `input_ids` / `pixel_values` are exactly what the policy sees.
The injected noise is drawn here and saved, so the split and the engines can be fed the
same draw.

    python reference.py --out ref.npz --embodiment robocasa_panda_omron --views 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from groot_split import load_policy


def build_inputs(ckpt: Path, policy, embodiment: str, views: int, task: str, seed: int):
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

    proc = Gr00tN1d6Processor.from_pretrained(str(ckpt))
    proc.eval()
    rng = np.random.default_rng(seed)
    raw = [rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(views)]
    imgs = [proc.eval_image_transform(image=im)["image"] for im in raw]
    stacked = np.stack([np.transpose(im, (2, 0, 1)) for im in imgs])        # [V,3,H,W]
    vlm = proc._apply_vlm_processing(stacked, task)
    batch = policy.collator([vlm])["inputs"]
    eid = json.loads((ckpt / "embodiment_id.json").read_text())[embodiment]
    state = torch.from_numpy(rng.standard_normal((1, 1, 128)).astype(np.float32))
    return raw, stacked, batch, eid, state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="nvidia/GR00T-N1.6-3B")
    ap.add_argument("--embodiment", default="robocasa_panda_omron")
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--task", default="pick up the red cube and place it in the bowl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    ckpt = Path(snapshot_download(a.checkpoint)) if not Path(a.checkpoint).is_dir() else Path(a.checkpoint)
    t0 = time.time()
    policy = load_policy(str(ckpt), torch.float32)
    print(f"loaded in {time.time() - t0:.0f} s")

    raw, stacked, batch, eid, state = build_inputs(ckpt, policy, a.embodiment, a.views, a.task, a.seed)
    for k, v in batch.items():
        shapes = [tuple(x.shape) for x in v] if isinstance(v, (list, tuple)) else tuple(v.shape)
        print(f"  {k}: {shapes}")

    g = torch.Generator().manual_seed(a.seed + 1)
    noise = torch.randn(1, policy.config.action_horizon, policy.config.max_action_dim, generator=g)

    head = policy.action_head
    real_randn = torch.randn
    torch.randn = lambda *s, **kw: noise.clone().to(kw.get("dtype") or noise.dtype)
    try:
        with torch.no_grad():
            bb = policy.backbone(policy.backbone.prepare_input(
                {k: batch[k] for k in ("input_ids", "attention_mask", "pixel_values")}))
            ai = head.prepare_input({"state": state, "embodiment_id": torch.tensor([eid])})
            out = head.get_action(bb, ai)
    finally:
        torch.randn = real_randn

    np.savez(
        a.out,
        raw=np.stack(raw), stacked=stacked,
        input_ids=batch["input_ids"].numpy(), attention_mask=batch["attention_mask"].numpy(),
        pixel_values=torch.cat(list(batch["pixel_values"])).numpy(),
        state=state.numpy(), noise=noise.numpy(), embodiment_id=np.array(eid),
        backbone_features=bb["backbone_features"].numpy(),
        image_mask=bb["image_mask"].numpy(),
        action_pred=out["action_pred"].numpy(),
        meta=np.array(json.dumps({"embodiment": a.embodiment, "views": a.views, "task": a.task,
                                  "seed": a.seed, "checkpoint": str(ckpt)})),
    )
    print("action_pred", tuple(out["action_pred"].shape), "saved", a.out)


if __name__ == "__main__":
    main()
