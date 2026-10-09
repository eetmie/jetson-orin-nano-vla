#!/usr/bin/env python3
# VENDORED from spark-projects @ c73dea0 — vla-onnx/groot/export_split_onnx17.py
"""Export GR00T N1.7 as split ONNX graphs for the 8 GB Orin, then a mixed-FP16 copy.

Graphs (static shapes, one per TensorRT engine; see groot17_split.py for the pipeline):

    vision_{k}  Qwen3-VL ViT chunks over `views` images; DeepStack chunks also emit ds_j,
                the last adds the merger
    llm_{k}     decoder-layer chunks over the padded sequence (+ DeepStack adds; no final norm)
    cond_{k}    vlln + VL self-attention chunks (+ state encoder in cond_0)
    time        the two sinusoidal encodings of t, kept FP32
    mod_{k}     every DiT block's AdaLN modulation from t alone; run at load, then unloaded
    kv_{k}      the cross-attention blocks' keys and values from vl; once per observation
    dit_{k}     one denoising step in chunks; the last adds the decoder and Euler update

The vision engines take one frame per view. The history frame is a second pass through
the same engines (or a cached result: an image's tokens depend on that image only).
The 151936 x 2048 token embedding ships as an FP16 .npy the runtime memory-maps.

    python export_split_onnx17.py --ref work/ref17.npz --out ~/bundles/groot-n17-base-split
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from export_split_onnx import SENSITIVE, dedupe_casts, dump, to_mixed_fp16, write_manifest
from groot17_split import (MASK_NEG, PATCH, Cond, Contract, LlmChunk, VisionChunk,
                           layer_plans, load_policy, mrope, n_params, vlm)
from groot_split import add_action_graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference17.py output: fixes the contract")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=448)
    ap.add_argument("--budget-m", type=float, default=105.0)
    ap.add_argument("--keep-fp32", action="store_true", help="also keep the FP32 graphs")
    a = ap.parse_args()

    r = np.load(a.ref)
    meta = json.loads(str(r["meta"]))
    policy = load_policy(meta["checkpoint"], meta["vlm_files"], torch.float32)
    m = vlm(policy)
    tok = m.config.image_token_id
    thw = r["image_grid_thw"]
    gh, gw = int(thw[0, 1]), int(thw[0, 2])
    c = Contract(meta["embodiment"], int(r["embodiment_id"]), meta["views"], meta["frames"],
                 (gh * PATCH, gw * PATCH), a.seq_len)
    ids = r["input_ids"]
    n_real = ids.shape[1]
    assert n_real <= c.seq_len and thw.shape[0] == c.images
    pad_id = m.config.text_config.eos_token_id
    ids_p = np.concatenate([ids[0], np.full(c.seq_len - n_real, pad_id)]).astype(np.int64)
    cos, sin = mrope(policy, torch.from_numpy(ids_p)[None], n_real, c)
    plans = layer_plans(policy, int(a.budget_m * 1e6))

    fp32 = a.out.parent / (a.out.name + ".fp32")
    if fp32.exists():
        shutil.rmtree(fp32)
    fp32.mkdir(parents=True)
    graphs = []
    S, V, P = c.seq_len, c.views, gh * gw
    D = m.config.text_config.hidden_size
    vd = m.config.vision_config.hidden_size
    n_ds = len(m.visual.deepstack_visual_indexes)

    def add(name, module, args, inputs, outputs):
        dump(module, args, fp32 / f"{name}.onnx", inputs, outputs)
        graphs.append({"name": name, "file": f"{name}.onnx", "inputs": inputs, "outputs": outputs,
                       "params_m": round(n_params(module) / 1e6, 1)})

    with torch.no_grad():
        j = 0
        for k, (lo, hi) in enumerate(plans["vision"]):
            mod = VisionChunk(policy, lo, hi, c).eval()
            x = torch.zeros(V, 3, *c.image_hw) if mod.first else torch.zeros(V, P, vd)
            ds = [f"ds_{j + i}" for i in range(len(mod.ds_at))]
            j += len(ds)
            add(f"vision_{k}", mod, (x,), ["pixel_values" if mod.first else "x"],
                ["vision_tokens" if mod.last else "x_out", *ds])
        for k, (lo, hi) in enumerate(plans["llm"]):
            mod = LlmChunk(policy, lo, hi, c, cos, sin, n_ds).eval()
            ds = [f"ds_{lo + i}" for i in mod.ds_layers]
            add(f"llm_{k}", mod, (torch.zeros(1, S, D), *[torch.zeros(1, S, D)] * len(ds)),
                ["h", *ds], ["features" if mod.last else "h_out"])
        for k, (lo, hi) in enumerate(plans["cond"]):
            mod = Cond(policy, lo, hi, c.embodiment_id).eval()
            if mod.first:
                add(f"cond_{k}", mod, (torch.zeros(1, S, D), torch.zeros(1, 1, S),
                                       torch.zeros(1, 1, c.state_dim)),
                    ["features", "pad_bias", "state"], ["vl", "state_features"])
            else:
                add(f"cond_{k}", mod, (torch.zeros(1, S, D), torch.zeros(1, 1, S)),
                    ["vl", "pad_bias"], ["vl_out"])
        add_action_graphs(add, policy, c, plans, S, D)

    emb = m.language_model.embed_tokens.weight.detach().to(torch.float16).numpy()
    np.save(fp32 / "embed_tokens.npy", emb)

    bundle = {
        "schema": "groot-n1.7-split/2",
        "model": "nvidia/GR00T-N1.7-3B",
        "checkpoint": meta["checkpoint"], "vlm_files": meta["vlm_files"],
        "embodiment": c.embodiment, "embodiment_id": c.embodiment_id,
        "views": V, "frames": c.frames, "video_delta_indices": meta["video_delta_indices"],
        "image_hw": list(c.image_hw), "tokens_per_image": c.tokens_per_image,
        "image_mean": 0.5, "image_std": 0.5,
        "seq_len": S, "prompt_tokens": n_real, "image_token": tok, "pad_token": pad_id,
        "task": meta["task"], "input_ids": ids_p.tolist(),
        "deepstack": n_ds, "mask_neg": MASK_NEG,
        "action_horizon": c.action_horizon, "action_dim": c.action_dim, "state_dim": c.state_dim,
        # The embodiment's real widths; the graphs run the padded 132.
        "state_used": meta["n_state"], "action_used": meta.get("n_action"),
        "timesteps": c.timesteps(),
        "embed_tokens": "embed_tokens.npy",
        "letterbox": True, "shortest_image_edge": 256, "crop_fraction": 0.95,
        # Stock-PyTorch FP32 outputs for seeded inputs (reference17.py); the runtime runs
        # the engines on the same inputs at load and fails closed on a mismatch.
        "fixture": {"file": "fixture.npz", "source": "stock PyTorch float32"},
        "chunks": {k: [list(p) for p in v] for k, v in plans.items()},
        "graphs": graphs,
    }
    (fp32 / "bundle.json").write_text(json.dumps(bundle, indent=2))

    print(f"\nmixed FP16 -> {a.out}  (FP32: {list(SENSITIVE)} + every RMSNorm node)")
    a.out.mkdir(parents=True, exist_ok=True)
    for g in graphs:
        src, dst = fp32 / g["file"], a.out / g["file"]
        if g["name"] == "time":
            shutil.copy2(src, dst)
        else:
            to_mixed_fp16(src, dst)
        g["size_mb"] = round(dst.stat().st_size / 1e6, 1)
    (a.out / "bundle.json").write_text(json.dumps(bundle, indent=2))
    print(f"  {sum(g['size_mb'] for g in graphs):.0f} MB of graphs")
    for g in graphs:
        n = dedupe_casts(a.out / g["file"])
        if n:
            print(f"  {g['name']}: dropped {n} duplicate Cast nodes")
    shutil.copy2(fp32 / "embed_tokens.npy", a.out / "embed_tokens.npy")
    shutil.copy2(a.ref, a.out / "fixture.npz")
    write_manifest(a.out)
    if not a.keep_fp32:
        shutil.rmtree(fp32)


if __name__ == "__main__":
    main()
