#!/usr/bin/env python3
# VENDORED from spark-projects @ c73dea0 — vla-onnx/groot/reference17.py
"""Run stock GR00T N1.7 on seeded synthetic inputs and save everything parity needs.

The frames go through the checkpoint's own eval image transform, Qwen3-VL chat template,
tokenizer and image processor, so the saved `input_ids` / `pixel_values` are exactly
what the policy sees. Raw frames are camera-sized (480x640) so the runtime's host
preprocessing can be checked against the stock transform. The injected noise is drawn
here and saved, so the split and the engines can be fed the same draw.

    python reference17.py --out work/ref17.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from groot17_split import load_policy


def build_inputs(ckpt: Path, vlm_files: str, embodiment: str, task: str, seed: int,
                 raw_hw: tuple[int, int]):
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    proc = Gr00tN1d7Processor.from_pretrained(str(ckpt))
    proc.eval()
    mc = proc.modality_configs[embodiment]
    views = len(mc["video"].modality_keys)
    frames = len(mc["video"].delta_indices)
    norm = proc.state_action_processor.norm_params[embodiment]
    n_state = sum(int(norm["state"][k]["dim"]) for k in mc["state"].modality_keys)
    n_action = sum(int(norm["action"][k]["dim"]) for k in mc["action"].modality_keys)
    rng = np.random.default_rng(seed)
    # (T, V) order, oldest frame first: how process_observation flattens (B, T, V, ...).
    raw = [rng.integers(0, 256, (*raw_hw, 3), dtype=np.uint8) for _ in range(frames * views)]
    imgs = [proc.eval_image_transform(image=im)["image"] for im in raw]
    stacked = np.stack([np.transpose(im, (2, 0, 1)) for im in imgs])        # [T*V,3,H,W]
    vlm_in = proc._apply_vlm_processing(stacked, task)
    batch = proc.collator([vlm_in])["inputs"]
    eid = proc.embodiment_id_mapping[embodiment]
    state = np.zeros((1, 1, proc.max_state_dim), np.float32)
    state[0, 0, :n_state] = rng.uniform(-1, 1, n_state)       # normalized, as the processor emits
    return (raw, stacked, batch, eid, torch.from_numpy(state), views, frames,
            {"n_state": n_state, "n_action": n_action}, proc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--vlm-files", default="nvidia/Cosmos-Reason2-2B",
                    help="config/tokenizer/processor source (gated HF repo, or a local dir)")
    ap.add_argument("--embodiment", default="xdof_relative_eef_relative_joint")
    ap.add_argument("--task", default="pick up the red cube and place it in the bowl")
    ap.add_argument("--raw-hw", type=int, nargs=2, default=[480, 640])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    ckpt = Path(a.checkpoint) if Path(a.checkpoint).is_dir() else Path(snapshot_download(a.checkpoint))
    t0 = time.time()
    policy = load_policy(str(ckpt), a.vlm_files, torch.float32)
    print(f"loaded in {time.time() - t0:.0f} s")

    raw, stacked, batch, eid, state, views, frames, dims, proc = build_inputs(
        ckpt, a.vlm_files, a.embodiment, a.task, a.seed, tuple(a.raw_hw))
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")

    g = torch.Generator().manual_seed(a.seed + 1)
    noise = torch.randn(1, policy.config.action_horizon, policy.config.max_action_dim, generator=g)

    head = policy.action_head
    real_randn = torch.randn
    torch.randn = lambda *s, **kw: noise.clone().to(kw.get("dtype") or noise.dtype)
    try:
        with torch.no_grad():
            bb = policy.backbone(policy.backbone.prepare_input(
                {k: batch[k] for k in ("input_ids", "attention_mask", "pixel_values",
                                       "image_grid_thw")}))
            ai = head.prepare_input({"state": state, "embodiment_id": torch.tensor([eid])})
            out = head.get_action(bb, ai)
    finally:
        torch.randn = real_randn

    np.savez(
        a.out,
        raw=np.stack(raw), stacked=stacked,
        input_ids=batch["input_ids"].numpy(), attention_mask=batch["attention_mask"].numpy(),
        pixel_values=batch["pixel_values"].numpy(), image_grid_thw=batch["image_grid_thw"].numpy(),
        state=state.numpy(), noise=noise.numpy(), embodiment_id=np.array(eid),
        backbone_features=bb["backbone_features"].numpy(),
        image_mask=bb["image_mask"].numpy(),
        action_pred=out["action_pred"].numpy(),
        meta=np.array(json.dumps({
            "embodiment": a.embodiment, "views": views, "frames": frames, **dims,
            "video_delta_indices": list(proc.modality_configs[a.embodiment]["video"].delta_indices),
            "task": a.task, "seed": a.seed, "checkpoint": str(ckpt), "vlm_files": a.vlm_files})),
    )
    print("action_pred", tuple(out["action_pred"].shape), "saved", a.out)


if __name__ == "__main__":
    main()
