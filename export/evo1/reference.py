#!/usr/bin/env python3
# Vendored from the author's fine-tuning pipeline: vla-onnx/evo1/emit_reference.py @ 268c686.
# Changed here: runs on the CPU by default, and also writes the token-embedding table
# (`embed_tokens.npy`) that the TensorRT runtime memory-maps.
"""Emit a deterministic native-LeRobot EVO1 reference fixture for split parity."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from export_split_onnx import _build_policy, write_manifest

ROOT = Path(__file__).resolve().parent


def _verify_manifest(bundle_dir: Path) -> None:
    import hashlib

    manifest = bundle_dir / "MANIFEST.sha256"
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = bundle_dir / relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"bundle identity mismatch: {relative}")


def _causal_mask(valid: np.ndarray) -> np.ndarray:
    length = valid.shape[1]
    causal = np.tril(np.ones((length, length), dtype=bool))
    allowed = causal[None, None, :, :] & valid[:, None, None, :].astype(bool)
    return np.where(allowed, 0.0, -10000.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=ROOT / "exports" / "split-bootstrap",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "models" / "InternVL3-1B-hf",
    )
    parser.add_argument("--task", default="move sand to the container")
    parser.add_argument("--noise-seed", type=int, default=12345)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    from lerobot.policies.evo1.internvl3_embedder import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        _batched_pixel_values,
    )

    bundle_dir = args.bundle_dir.resolve()
    _verify_manifest(bundle_dir)
    bundle = json.loads((bundle_dir / "bundle.json").read_text())
    # Trained bundles get a fixture too -- that is the whole point of one, since EVO1
    # has no PyTorch backend on the Orin to compare against. What must not happen is a
    # fixture built from a DIFFERENT policy than the graphs: a random head measured
    # against random-head graphs passes, and so does a trained head against trained
    # graphs, but crossing them silently certifies nothing. So the checkpoint is read
    # from the bundle rather than passed in.
    ckpt = (bundle.get("checkpoint") or {}).get("path")
    if bool(ckpt) == bool(bundle.get("random_action_head")):
        raise ValueError(
            "bundle disagrees with itself: random_action_head="
            f"{bundle.get('random_action_head')} but checkpoint={ckpt!r}")
    views = int(bundle["max_views"])

    start = time.time()
    policy = _build_policy(
        args.base.resolve(),
        int(bundle["seed"]),
        int(bundle["seq_len"]),
        views,
        Path(ckpt) if ckpt else None,
    )
    policy.to(args.device).eval()
    owner = policy.model.embedder.model
    tokenizer = policy.model.embedder.tokenizer

    # One draw per view, from a single stream, so a 2-view fixture's first camera is
    # NOT the same image a 1-view fixture would have used. Reusing one image across
    # views would hide a bug that swaps or drops a view.
    image_rng = np.random.default_rng(20260902)
    raw_images = [
        image_rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        for _ in range(views)
    ]
    images = [
        torch.from_numpy(raw)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(args.device, dtype=torch.float32)
        / 255.0
        for raw in raw_images
    ]
    image = images[0]
    mean = torch.tensor(IMAGENET_MEAN, device=args.device, dtype=torch.float32)
    std = torch.tensor(IMAGENET_STD, device=args.device, dtype=torch.float32)
    pixel_values = _batched_pixel_values(
        images,
        max_views=views,
        image_size=int(bundle["image_size"]),
        mean=mean,
        std=std,
        dtype=torch.float32,
        device=args.device,
    )

    # One tile per view: the prompt must declare exactly the views the pixel stack
    # carries, or the image-token count below will not match.
    prompt = policy.model.embedder._build_multimodal_prompts(
        [[1] * views],
        [args.task],
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=int(bundle["seq_len"]),
    ).to(args.device)
    input_ids = encoded["input_ids"]
    context_mask = encoded["attention_mask"].to(torch.bool)
    image_tokens = int(
        (input_ids == int(bundle["image_token_id"])).sum().item()
    )
    expected_tokens = views * int(bundle["image_seq_length"])
    if image_tokens != expected_tokens:
        raise ValueError(
            f"prompt has {image_tokens} image tokens, expected {expected_tokens} "
            f"({views} views x {bundle['image_seq_length']})"
        )

    with torch.no_grad():
        vision_start = time.time()
        image_features = owner.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
            return_dict=True,
        ).pooler_output
        vision_s = time.time() - vision_start

        token_embeddings = owner.language_model.embed_tokens(input_ids)
        image_mask = (input_ids == int(bundle["image_token_id"])).unsqueeze(-1)
        image_mask = image_mask.expand_as(token_embeddings)
        merged = token_embeddings.masked_scatter(
            image_mask,
            image_features.to(token_embeddings.dtype),
        )
        language_start = time.time()
        fused_tokens = owner.language_model(
            attention_mask=encoded["attention_mask"],
            inputs_embeds=merged,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        language_s = time.time() - language_start

        state = torch.linspace(
            -0.5,
            0.5,
            int(bundle["max_state_dim"]),
            device=args.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        action_mask = torch.ones(
            1,
            int(bundle["max_action_dim"]),
            device=args.device,
            dtype=torch.bool,
        )
        embodiment = torch.zeros(1, device=args.device, dtype=torch.long)

        torch.manual_seed(args.noise_seed)
        initial_noise = torch.rand(
            1,
            int(bundle["chunk_size"]) * int(bundle["max_action_dim"]),
            device=args.device,
            dtype=fused_tokens.dtype,
        )
        initial_noise = initial_noise.mul(2).sub(1).view(
            1,
            int(bundle["chunk_size"]),
            int(bundle["max_action_dim"]),
        )
        torch.manual_seed(args.noise_seed)
        action_start = time.time()
        expected_action = policy.model.action_head.get_action(
            fused_tokens,
            state=state,
            action_mask=action_mask,
            embodiment_id=embodiment,
            context_mask=context_mask,
        ).view(
            1,
            int(bundle["chunk_size"]),
            int(bundle["max_action_dim"]),
        )
        action_s = time.time() - action_start

    table = owner.language_model.embed_tokens.weight.detach().float().cpu().numpy()
    np.save(bundle_dir / "embed_tokens.npy", table)
    bundle["embed_tokens"] = "embed_tokens.npy"

    fixture_path = bundle_dir / "parity_fixture.npz"
    np.savez_compressed(
        fixture_path,
        # raw_image stays the FIRST view's array so a one-view fixture is byte-identical
        # to what this emitter produced before multi-view; raw_images carries the full
        # stack. A consumer that only knows raw_image therefore still works on a
        # one-view bundle and cannot silently half-read a two-view one.
        raw_image=raw_images[0],
        raw_images=np.stack(raw_images),
        pixel_values=pixel_values.cpu().numpy().astype(np.float32),
        input_ids=input_ids.cpu().numpy().astype(np.int64),
        context_mask=context_mask.cpu().numpy().astype(bool),
        causal_mask=_causal_mask(context_mask.cpu().numpy()),
        state=state.cpu().numpy().astype(np.float32),
        initial_noise=initial_noise.cpu().numpy().astype(np.float32),
        expected_vision=image_features.cpu().numpy().astype(np.float32),
        expected_fused=fused_tokens.cpu().numpy().astype(np.float32),
        expected_action=expected_action.cpu().numpy().astype(np.float32),
    )
    bundle["fixture"] = {
        "file": fixture_path.name,
        "task": args.task,
        "noise_seed": args.noise_seed,
        "reference": f"native LeRobot 0.6.1 EVO1, FP32 {args.device}",
    }
    bundle["reference_timings_s"] = {
        "vision": round(vision_s, 4),
        "language": round(language_s, 4),
        "action_32_steps": round(action_s, 4),
        "total_with_load": round(time.time() - start, 4),
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    files = write_manifest(bundle_dir)
    print(f"reference fixture: {fixture_path}")
    print(
        f"native timings: vision={vision_s:.3f}s language={language_s:.3f}s "
        f"action32={action_s:.3f}s"
    )
    print(
        f"expected action range: {expected_action.min().item():.6f} "
        f"to {expected_action.max().item():.6f}"
    )
    print(f"rewrote MANIFEST.sha256 ({files} files)")


if __name__ == "__main__":
    main()
