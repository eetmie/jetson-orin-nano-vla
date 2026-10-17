#!/usr/bin/env python3
"""Export a LeRobot π0.5 checkpoint to the compact pi0.5 split bundle the board refits.

    python export_split.py --checkpoint DIR --out DIR [--views N] [--task T]... [--fixtures 3]

Stock fixtures run LeRobot's own pipeline: the checkpoint's preprocessor (quantile
normalization, the `Task: ..., State: <bins>;\\nAction: ` prompt, PaliGemma tokenizer),
PI05Policy in FP32 on CUDA with injected noise, and its postprocessor. The FP16 split
reference is the layout the board runs: the real cameras, then the prompt's real tokens,
then zeros to a static capacity. Masked camera slots and prompt padding are never
computed; both are exact in FP32 (scratch tests of 2026-10-11).

The prompt changes with the state every step, so the bundle carries each task's token
head and the token pieces of the state digits; the board assembles the prompt without a
tokenizer. The export checks that assembly against the tokenizer for every bin.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layers import (MASK_NEG, ActionBlock, ActionInput, ActionOutput, GemmaBlock, KVBlock,  # noqa: E402
                    VisionBlock, VisionStem, VisionTail, accumulation, array, metric, pack, sha)

IMAGE_TOKENS = 256
HIDDEN = 2048
STATE_PIECES = "0123456789"
# Worst case per state value: the separator piece plus three digits.
TOKENS_PER_STATE = 4
DEFAULT_TASK = "pick up the red block"


def quantile_stats(pipeline, key: str, feature_type) -> tuple[dict | None, float]:
    """(q01/q99 of `key`, eps) from a pipeline's normalizer step; None when it has none."""
    from lerobot.configs.types import NormalizationMode

    for step in pipeline.steps:
        stats = getattr(step, "_tensor_stats", None)
        if stats is None or not hasattr(step, "norm_map"):
            continue
        if key not in stats:
            return None, float(step.eps)
        mode = step.norm_map.get(feature_type)
        if mode != NormalizationMode.QUANTILES:
            raise SystemExit(f"{key}: normalization {mode} is not supported (pi0.5 uses QUANTILES)")
        s = stats[key]
        return ({"q01": s["q01"].float().cpu().numpy().ravel().tolist(),
                 "q99": s["q99"].float().cpu().numpy().ravel().tolist()}, float(step.eps))
    return None, 1e-8


def normalize(x: np.ndarray, stats: dict | None, eps: float) -> np.ndarray:
    """LeRobot QUANTILES, FP32, in its operation order."""
    if stats is None:
        return x.astype(np.float32)
    q01, q99 = np.float32(stats["q01"]), np.float32(stats["q99"])
    denom = q99 - q01
    denom = np.where(denom == 0, np.float32(eps), denom)
    return np.float32(2.0) * (x.astype(np.float32) - q01) / denom - np.float32(1.0)


def unnormalize(x: np.ndarray, stats: dict | None, eps: float) -> np.ndarray:
    if stats is None:
        return x.astype(np.float32)
    q01, q99 = np.float32(stats["q01"]), np.float32(stats["q99"])
    denom = q99 - q01
    denom = np.where(denom == 0, np.float32(eps), denom)
    return (x.astype(np.float32) + np.float32(1.0)) * denom / np.float32(2.0) + q01


def state_bins(state_norm: np.ndarray) -> np.ndarray:
    return np.digitize(state_norm, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1


class Prompt:
    """The tokenizer's output for `Task: T, State: b b ...;\\nAction: `, from pieces."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        # A bin is " <digits>"; the one negative bin, -1, merges its space and sign: "▁-".
        self.sep = tokenizer.convert_tokens_to_ids("▁")
        self.neg = tokenizer.convert_tokens_to_ids("▁-")
        self.pieces = {c: tokenizer.convert_tokens_to_ids(c) for c in STATE_PIECES}
        self.tail = tokenizer(";\nAction: ", add_special_tokens=False)["input_ids"]
        unknown = tokenizer.unk_token_id
        assert unknown not in (self.sep, self.neg, *self.pieces.values()), "state pieces are not single tokens"

    @staticmethod
    def clean(task: str) -> str:
        return task.strip().replace("_", " ").replace("\n", " ")

    def head(self, task: str) -> list[int]:
        return self.tok(f"Task: {self.clean(task)}, State:")["input_ids"]

    def compose(self, head: list[int], bins) -> list[int]:
        ids = list(head)
        for v in bins:
            v = int(v)
            ids += [self.neg if v < 0 else self.sep] + [self.pieces[c] for c in str(abs(v))]
        return ids + self.tail

    def full(self, task: str, bins) -> list[int]:
        text = f"Task: {self.clean(task)}, State: {' '.join(map(str, bins))};\nAction: "
        return self.tok(text)["input_ids"]

    def verify(self, task: str, state_dim: int, rng) -> int:
        head, n = self.head(task), 0
        for v in range(-1, 256):
            assert self.compose(head, [v]) == self.full(task, [v]), (task, v)
            n += 1
        for _ in range(300):
            bins = rng.integers(-1, 256, state_dim)
            assert self.compose(head, bins) == self.full(task, bins), (task, bins[:6])
            n += 1
        return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", type=int, default=None, help="real cameras (default: all the checkpoint declares)")
    ap.add_argument("--task", action="append", default=None)
    ap.add_argument("--fixtures", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--openpi-spec", type=Path, help="resolved specification from the host inspector")
    a = ap.parse_args()
    if a.openpi_spec:
        from openpi_provider import export as export_openpi
        export_openpi(a, export_loaded)
        return


    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType
    from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    from vla_common.bundle import write_manifest
    from vla_common.bundle.provenance import package_versions

    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixtures").mkdir()
    tasks = a.task or [DEFAULT_TASK]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    ckpt = str(a.checkpoint.resolve())
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    if cfg.type != "pi05":
        ap.error(f"not a LeRobot pi05 checkpoint (type {cfg.type!r})")
    cfg.device = "cuda"
    pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    keys = list(cfg.image_features)
    views = a.views or len(keys)
    if not 1 <= views <= len(keys):
        ap.error(f"--views must be 1..{len(keys)}")
    keys = keys[:views]
    state_dim = cfg.input_features["observation.state"].shape[0]
    robot_dim = cfg.output_features["action"].shape[0]
    horizon, width, steps = cfg.chunk_size, cfg.max_action_dim, cfg.num_inference_steps
    state_stats, eps = quantile_stats(pre, "observation.state", FeatureType.STATE)
    action_stats, eps_out = quantile_stats(post, "action", FeatureType.ACTION)

    tok_step = next(s for s in pre.steps if hasattr(s, "input_tokenizer"))
    prompt = Prompt(tok_step.input_tokenizer)
    rng = np.random.default_rng(a.seed)
    checked = sum(prompt.verify(t, state_dim, rng) for t in tasks)
    print(f"prompt assembly == tokenizer on {checked} prompts", flush=True)
    heads = {t: prompt.head(t) for t in tasks}
    text_capacity = max(len(h) for h in heads.values()) + TOKENS_PER_STATE * state_dim + len(prompt.tail)
    if text_capacity > tok_step.max_length:
        ap.error(f"worst-case prompt {text_capacity} tokens exceeds the tokenizer's {tok_step.max_length}; "
                 "LeRobot would truncate it")
    capacity = views * IMAGE_TOKENS + text_capacity
    print(f"views {views}, state {state_dim}, horizon {horizon}: prefix capacity {capacity} "
          f"({text_capacity} prompt)", flush=True)
    print("loading the stock policy in FP32 on CUDA", flush=True)
    policy = PI05Policy.from_pretrained(ckpt, config=cfg).to("cuda").float().eval()
    model = policy.model

    # ── stock fixtures ───────────────────────────────────────────────────────
    fixtures = []
    for i in range(a.fixtures):
        task = tasks[i % len(tasks)]
        raw = np.stack([rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in keys])
        # The last fixture reaches past the quantiles so bins -1 and 255 are exercised.
        widen = 1.5 if i == a.fixtures - 1 else 1.0
        if state_stats is None:
            state = rng.uniform(-widen, widen, state_dim).astype(np.float32)
        else:
            lo, hi = np.float32(state_stats["q01"]), np.float32(state_stats["q99"])
            mid, half = (lo + hi) / 2, (hi - lo) / 2 * widen
            state = rng.uniform(mid - half, mid + half).astype(np.float32)
        noise = rng.standard_normal((1, horizon, width)).astype(np.float32)
        obs = {k: torch.from_numpy(im.transpose(2, 0, 1).copy()).float() / 255.0 for k, im in zip(keys, raw)}
        obs["observation.state"] = torch.from_numpy(state)
        obs["task"] = task
        batch = pre(obs)
        images, img_masks = policy._preprocess_images(batch)
        tokens, tmask = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        with torch.no_grad():
            act = model.sample_actions(images, img_masks, tokens, tmask,
                                       noise=torch.from_numpy(noise).cuda())
        robot_stock = post(act[:, :, :robot_dim].clone())
        robot_stock = robot_stock.float().cpu().numpy() if torch.is_tensor(robot_stock) else np.asarray(robot_stock)
        stock = array(act.float())
        ids = array(tokens[0][tmask[0].bool()]).astype(np.int64)
        state_norm = normalize(state, state_stats, eps)
        bins = state_bins(state_norm)
        assert prompt.compose(heads[task], bins) == ids.tolist(), ("board prompt differs", i)
        mine = unnormalize(stock[0, :, :robot_dim], action_stats, eps_out)
        assert np.allclose(mine, robot_stock.reshape(mine.shape), rtol=0, atol=1e-5), "unnormalize differs"
        pixels = np.stack([array(im[0]) for im, m in zip(images, img_masks) if bool(m.item())])
        fixtures.append({"task": task, "raw": raw, "pixels": pixels.astype(np.float32), "state": state,
                         "state_norm": state_norm, "bins": bins.astype(np.int64), "tokens": ids,
                         "noise": noise, "stock_actions": stock,
                         "stock_robot": robot_stock.reshape(horizon, robot_dim).astype(np.float32)})
        print(f"stock fixture {i}: {task!r}, {len(ids)} prompt tokens, bins {bins.min()}..{bins.max()}",
              flush=True)

    metadata = {
        "policy": "pi05_lerobot", "prompt_mode": "state", "schedule": "lerobot",
        "image_keys": keys, "valid_cameras": views, "state_dim": state_dim,
        "action_horizon": horizon, "action_dim": width, "robot_action_dim": robot_dim, "num_steps": steps,
        "prefix_tokens": capacity, "prompt_capacity": text_capacity,
        "min_period": cfg.min_period, "max_period": cfg.max_period,
        "normalization": {"mode": "QUANTILES", "implementation": "lerobot", "state": state_stats,
                          "state_eps": eps, "action": action_stats, "action_eps": eps_out},
        "image_preprocess": "LeRobot: uint8/255, bilinear without antialias, centered padding, x*2-1",
        "tokenizer": {"name": tok_step.tokenizer_name, "max_length": tok_step.max_length,
                      "sep": prompt.sep, "neg": prompt.neg, "pieces": prompt.pieces, "tail": prompt.tail,
                      "assembly_checked_prompts": checked},
        "tasks": {t: {"head": h} for t, h in heads.items()},
    }
    export_loaded(a, model, fixtures, metadata, create_sinusoidal_pos_embedding,
                  lambda x: unnormalize(x[:, :robot_dim], action_stats, eps_out))


def export_loaded(a, model, fixtures, metadata, sinusoidal, postprocess):
    """Shared split/template export; each framework supplies its own stock-policy fixtures."""
    from vla_common.bundle import write_manifest
    from vla_common.bundle.provenance import package_versions

    out = a.out.resolve()
    ckpt = str(a.checkpoint.resolve())
    keys, views = metadata["image_keys"], metadata["valid_cameras"]
    state_dim, robot_dim = metadata["state_dim"], metadata["robot_action_dim"]
    horizon, width, steps = metadata["action_horizon"], metadata["action_dim"], metadata["num_steps"]
    capacity, text_capacity = metadata["prefix_tokens"], metadata["prompt_capacity"]
    # ── FP16 split model ─────────────────────────────────────────────────────
    core = model.paligemma_with_expert.paligemma.model
    language, vision = core.language_model, core.vision_tower.vision_model
    expert = model.paligemma_with_expert.gemma_expert.model
    rope_frequencies = [m.rotary_emb.inv_freq.clone().float() for m in (language, expert)]
    model.half()
    for module, frequencies in zip((language, expert), rope_frequencies):
        module.rotary_emb.inv_freq = frequencies
        module.rotary_emb.original_inv_freq = frequencies
        module.config._attn_implementation = "eager"
    vision.config._attn_implementation = "eager"

    stem = VisionStem(vision.embeddings)
    vision_blocks = [VisionBlock(layer) for layer in vision.encoder.layers]
    # The language residual stream is FP32 (see GemmaBlock); everything else stays FP16.
    tail = VisionTail(vision.post_layernorm, core.multi_modal_projector, fp32_out=True)
    language_blocks = [GemmaBlock(layer, residual32=True) for layer in language.layers[:-1]]
    last_kv = KVBlock(language.layers[-1], residual32=True)
    last_full = GemmaBlock(language.layers[-1], residual32=True)
    action_input = ActionInput(model.action_in_proj)
    action_blocks = [ActionBlock(layer) for layer in expert.layers]
    action_output = ActionOutput(expert.norm, model.action_out_proj)
    n_vision, n_lang, n_action = len(vision_blocks), len(language.layers), len(action_blocks)
    expert_width = model.action_in_proj.out_features

    # Keep each framework's FP32 timestep schedule and rounding order.
    dt = -1.0 / steps
    if metadata.get("schedule") == "openpi":
        time = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        step_dt = torch.tensor(dt, dtype=torch.float32, device="cuda")
        times = []
        for _ in range(steps):
            times.append(time.clone())
            time += step_dt
        times = torch.stack(times)
    else:
        times = torch.tensor([1.0 + s * dt for s in range(steps)], dtype=torch.float32, device="cuda")
    sites = [(expert.layers[i], norm) for i in range(n_action)
             for norm in ("input_layernorm", "post_attention_layernorm")]

    def condition(t):
        emb = sinusoidal(t, expert_width, metadata.get("min_period", 4e-3), metadata.get("max_period", 4.0),
                                              device=t.device).float().half()
        return F.silu(model.time_mlp_out(F.silu(model.time_mlp_in(emb))))

    with torch.inference_mode():
        conds = [condition(times[s:s + 1]) for s in range(steps)]
        styles = torch.empty((steps, len(sites) + 1, 3 * expert_width), dtype=torch.float16, device="cuda")
        for s in range(steps):
            for j, (layer, norm) in enumerate(sites):
                styles[s, j] = getattr(layer, norm).dense(conds[s])[0]
            styles[s, -1] = expert.norm.dense(conds[s])[0]
        positions = torch.arange(-1, 2048, device="cuda")[None]
        dummy = torch.empty((1, positions.shape[1], HIDDEN), dtype=torch.float16, device="cuda")
        rope_cos, rope_sin = language.rotary_emb(dummy, positions)
    del dummy
    embed16 = language.embed_tokens.weight.detach().half()
    scale = torch.tensor(HIDDEN ** 0.5, dtype=torch.float32, device="cuda")

    def split_run(fx, capture=None, live=False):
        with torch.inference_mode():
            parts = []
            for v in range(len(fx["pixels"])):
                image = torch.from_numpy(fx["pixels"][v:v + 1]).cuda().half()
                if capture is not None and v == 0:
                    capture["stem"] = (image,)
                hidden = stem(image)
                for i, block in enumerate(vision_blocks):
                    if capture is not None and v == 0:
                        capture[f"vision_{i:02d}"] = (hidden.clone(),)
                    hidden = block(hidden)
                if capture is not None and v == 0:
                    capture["vision_tail"] = (hidden.clone(),)
                parts.append(tail(hidden))
            ids = torch.from_numpy(fx["tokens"]).cuda()
            # Rounded to FP16 as the embedding is, then widened for the FP32 stream.
            rows = (embed16[ids].float() * scale).half().float()[None]
            n_valid = len(parts) * IMAGE_TOKENS + rows.shape[1]
            assert n_valid <= capacity, ("prefix capacity exceeded", n_valid, capacity)
            zeros = torch.zeros((1, capacity - n_valid, HIDDEN), dtype=torch.float32, device="cuda")
            hidden = torch.cat(parts + [rows, zeros], dim=1)
            pad = torch.arange(capacity, device="cuda")[None] < n_valid
            mask = torch.where(pad[:, None, :, None] & pad[:, None, None, :], 0.0, MASK_NEG)
            pos = torch.cumsum(pad, dim=-1) - 1
            cos, sin = rope_cos[:, pos[0] + 1], rope_sin[:, pos[0] + 1]
            kv = []
            for i, block in enumerate(language_blocks):
                if capture is not None:
                    capture[f"language_{i:02d}"] = (hidden.clone(), mask, cos, sin)
                hidden, k, v = block(hidden, mask, cos, sin)
                assert torch.isfinite(k).all() and torch.isfinite(v).all(), ("nonfinite K/V", i)
                peak = float(hidden.abs().max())
                ranges.append(peak)
                assert np.isfinite(peak), ("nonfinite language block output", i)
                kv.append((k, v))
            # The last layer: only its K/V (KVBlock); checked against the full layer once.
            if capture is not None:
                capture[f"language_{n_lang - 1:02d}"] = (hidden.clone(), cos, sin)
                _, k_full, v_full = last_full(hidden, mask, cos, sin)
            k, v = last_kv(hidden, cos, sin)
            if capture is not None:
                assert torch.equal(k, k_full) and torch.equal(v, v_full), "K/V-only last layer differs"
            assert torch.isfinite(k).all() and torch.isfinite(v).all(), ("nonfinite K/V", n_lang - 1)
            kv.append((k, v))
            spos = n_valid + torch.arange(horizon, device="cuda")
            s_cos, s_sin = rope_cos[:, spos + 1], rope_sin[:, spos + 1]
            s_valid = torch.cat((pad[:, None, :].expand(1, horizon, -1),
                                 torch.ones((1, horizon, horizon), dtype=torch.bool, device="cuda")), dim=-1)
            s_mask = torch.where(s_valid[:, None], 0.0, MASK_NEG)
            actions = torch.from_numpy(fx["noise"]).cuda()
            for s in range(steps):
                if capture is not None and s == 0:
                    capture["action_input"] = (actions.clone(),)
                hidden = action_input(actions)
                for i, block in enumerate(action_blocks):
                    mod_in, mod_post = styles[s, 2 * i][None, None], styles[s, 2 * i + 1][None, None]
                    if live:
                        mod_in = expert.layers[i].input_layernorm.dense(conds[s])[:, None]
                        mod_post = expert.layers[i].post_attention_layernorm.dense(conds[s])[:, None]
                    args = (hidden, s_mask, s_cos, s_sin, *kv[i], mod_in, mod_post)
                    if capture is not None and s == 0:
                        capture[f"action_{i:02d}"] = tuple(x.clone() for x in args)
                    hidden = block(*args)
                    assert torch.isfinite(hidden).all(), ("nonfinite FP16 action block", s, i)
                mod_final = styles[s, -1][None, None]
                if live:
                    mod_final = expert.norm.dense(conds[s])[:, None]
                if capture is not None and s == 0:
                    capture["action_output"] = (hidden.clone(), mod_final)
                velocity = action_output(hidden, mod_final)
                assert torch.isfinite(velocity).all(), ("nonfinite velocity", s)
                actions = actions + dt * velocity
            return array(actions)

    capture, comparisons, ranges = {}, [], []
    for i, fx in enumerate(fixtures):
        ranges.clear()
        fp16 = split_run(fx, capture=capture if i == 0 else None)
        peaks = [x if np.isfinite(x) else "inf" for x in ranges]
        if i == 0:
            assert np.array_equal(fp16, split_run(fx, live=True)), "cached and live AdaRMS modulation differ"
        robot = postprocess(fp16[0])
        c = {"fixture": i, "task": fx["task"], "prompt_tokens": int(len(fx["tokens"])),
             "language_output_max_abs_fp16": peaks,
             "fp16_vs_stock_fp32": metric(fp16, fx["stock_actions"]),
             "fp16_vs_stock_fp32_robot": metric(robot, fx["stock_robot"])}
        print(f"fixture {i} FP16 language output peaks: {peaks}", flush=True)
        assert c["fp16_vs_stock_fp32"]["finite"] and c["fp16_vs_stock_fp32"]["cosine"] >= 0.999, c
        comparisons.append(c)
        fx["fp16_actions"], fx["fp16_robot"] = fp16, robot
        np.savez_compressed(out / "fixtures" / f"fixture_{i}.npz",
                            **{k: (np.array(v) if k == "task" else v) for k, v in fx.items()})
        print("FP16 split vs stock", json.dumps(c["fp16_vs_stock_fp32"]), flush=True)

    # ── templates and per-component weights ──────────────────────────────────
    groups = [("stem", stem, "stem", ["image"], ["hidden"]),
              *[(f"vision_{i:02d}", b, "vision", ["hidden"], ["output"]) for i, b in enumerate(vision_blocks)],
              ("vision_tail", tail, "vision_tail", ["hidden"], ["output"]),
              *[(f"language_{i:02d}", b, "language", ["hidden", "mask", "cos", "sin"], ["output", "key", "value"])
                for i, b in enumerate(language_blocks)],
              (f"language_{n_lang - 1:02d}", last_kv, "language_kv", ["hidden", "cos", "sin"], ["key", "value"]),
              ("action_input", action_input, "action_input", ["actions"], ["hidden"]),
              *[(f"action_{i:02d}", b, "action",
                 ["hidden", "mask", "cos", "sin", "prefix_k", "prefix_v", "mod_in", "mod_post"], ["output"])
                for i, b in enumerate(action_blocks)],
              ("action_output", action_output, "action_output", ["hidden", "mod_final"], ["velocity"])]
    components, seen = [], set()
    temporary = out / "temporary_export.onnx"
    for label, module, kind, names, outputs in groups:
        with torch.no_grad():
            torch.onnx.export(module, capture[label], str(temporary), dynamo=False, opset_version=17,
                              input_names=names, output_names=outputs, do_constant_folding=True)
        graph = onnx.load(str(temporary))
        changed = accumulation(graph)
        record = pack(out, graph, kind, label, template=kind not in seen)
        seen.add(kind)
        components.append({"label": label, "kind": kind, "inputs": names, "outputs": outputs,
                           "raw_weights_bytes": record["initializer_bytes"], "accumulation_ops": changed,
                           "structure_sha256": record["structure_sha256"]})
        print(f"exported {label} ({record['initializer_bytes'] / 2**20:.1f} MiB)", flush=True)
        del graph
    temporary.unlink()

    np.save(out / "embedding_fp16.npy", array(embed16))
    np.save(out / "rope_cos_fp16.npy", array(rope_cos[0]))
    np.save(out / "rope_sin_fp16.npy", array(rope_sin[0]))
    conditioning = f"conditioning_{steps}step_fp16.npy"
    np.save(out / conditioning, array(styles))
    (out / "norm_stats.json").write_text(json.dumps(metadata["normalization"], indent=2) + "\n")
    weights = a.checkpoint.resolve() / "model.safetensors"
    snapshot = a.checkpoint.resolve() / ".vla-hub-snapshot.json"
    bundle = {
        "schema": "pi05-split/1", **{k: v for k, v in metadata.items() if k != "normalization"}, "complete": True, "layout": "compact",
        "source": {"checkpoint": json.loads(snapshot.read_text()) if snapshot.exists() else ckpt,
                   "model_sha256": sha(weights), "packages": package_versions(("lerobot", "openpi", "torch", "transformers"))},
        "image_keys": keys, "image_slots": views, "valid_cameras": views,
        "prefix_tokens": capacity, "prompt_capacity": text_capacity,
        "action_horizon": horizon, "action_dim": width, "robot_action_dim": robot_dim,
        "state_dim": state_dim, "num_steps": steps, "conditioning": conditioning,
        "vision_layers": n_vision, "language_layers": n_lang, "action_layers": n_action,
        "dtype": "FP16 weights and hidden states, FP32 language residual stream, "
                 "FP32 normalization/accumulation/Euler",
        "language_residual": "float32",
        "reference_comparisons": comparisons,
        "embedding_sha256": sha(out / "embedding_fp16.npy"),
        "components": components,
        "raw_compute_weights_bytes": sum(c["raw_weights_bytes"] for c in components),
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    write_manifest(out)
    print(f"complete: {out} ({bundle['raw_compute_weights_bytes'] / 2**30:.2f} GiB compute weights)", flush=True)


if __name__ == "__main__":
    main()
