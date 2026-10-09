#!/usr/bin/env python3
# Vendored from the author's fine-tuning pipeline: vla-onnx/evo1/export_split_onnx.py @ 268c686.
"""Export a deterministic EVO1 bootstrap policy as Orin-sized split ONNX graphs.

This first bundle validates export, numerical parity, TensorRT build memory, and runtime
mechanics. EVO1 has no generic pretrained robot action head, so the action head is
initialized deterministically from --seed and the bundle is marked non-deployable.
Replace it with a trained LeRobot checkpoint before robot control.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

from vla_common.bundle import validate_onnx, write_manifest
from vla_common.bundle.provenance import package_version

ROOT = Path(__file__).resolve().parent
FAMILIES = ("vision", "language", "action")
VISION_SPLITS = (7, 7, 7, 3)
LANGUAGE_SPLITS = (6, 6, 2)
IMAGE_SEQ_LENGTH = 256
BASE_REVISION = "014c0583a0d4bedf29fbe2dbff4f865eb998e171"
MODEL_ID = "OpenGVLab/InternVL3-1B-hf"


def _build_policy(base: Path, seed: int, seq_len: int, max_views: int = 1,
                  checkpoint: Path | None = None):
    """The policy to trace: a trained LeRobot checkpoint, or a fresh bootstrap head.

    `max_views` and `max_text_length` are safe to set on a trained checkpoint because
    neither sizes a weight. `max_views` only controls how far the view stack is padded
    and how wide the image mask is, and `max_text_length` is a tokenizer truncation
    length -- so exporting LIBERO's 2 real cameras at max_views=2 rather than the
    checkpoint's max_views=3 drops 256 masked positions from every sequence and changes
    nothing about the two real views.
    """
    import torch

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.evo1.configuration_evo1 import Evo1Config
    from lerobot.policies.evo1.modeling_evo1 import Evo1Policy
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    torch.manual_seed(seed)

    if checkpoint is not None:
        policy = Evo1Policy.from_pretrained(str(checkpoint))
        config = policy.config
        # The checkpoint was trained against the Hub id; point it at the verified local
        # base so the export never reaches the network mid-trace.
        config.vlm_model_name = str(base)
        config.device = "cpu"
        config.vlm_dtype = "float32"
        config.use_amp = False
        config.use_flash_attn = False
        config.enable_gradient_checkpointing = False
        config.max_views = max_views
        config.max_text_length = seq_len
        return policy.to("cpu").float().eval()

    config = Evo1Config(
        device="cpu",
        training_stage="stage1",
        vlm_model_name=str(base),
        vlm_dtype="float32",
        use_amp=False,
        use_flash_attn=False,
        enable_gradient_checkpointing=False,
        max_views=max_views,
        max_text_length=seq_len,
        input_features={
            **{
                f"{OBS_IMAGES}.image{'' if i == 0 else i + 1}": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 448, 448))
                for i in range(max_views)
            },
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(24,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(24,)),
        },
    )
    return Evo1Policy(config).eval()


def _state_elements(module) -> int:
    return sum(tensor.numel() for tensor in module.state_dict().values())



def _build_wrappers():
    import torch
    import torch.nn.functional as functional
    from torch import nn

    class VisionChunk(nn.Module):
        def __init__(self, embeddings, layers, projector=None):
            super().__init__()
            self.embeddings = embeddings
            self.layers = nn.ModuleList(layers)
            self.projector = projector

        def forward(self, value):
            hidden = self.embeddings(value) if self.embeddings is not None else value
            for layer in self.layers:
                hidden = layer(hidden)
            if self.projector is None:
                return hidden

            hidden = hidden[:, 1:, :]
            batch = hidden.shape[0]
            hidden = hidden.reshape(batch, 32, 32, 1024)
            hidden = hidden.view(batch, 32, 16, 2048)
            hidden = hidden.permute(0, 2, 1, 3).contiguous()
            hidden = hidden.view(batch, 16, 16, 4096)
            hidden = hidden.permute(0, 2, 1, 3).contiguous()
            hidden = hidden.reshape(batch, 256, 4096)
            return self.projector(hidden)

    class TokenEmbedding(nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embedding = embedding

        def forward(self, input_ids):
            return self.embedding(input_ids)

    class LanguageChunk(nn.Module):
        def __init__(self, layers, cos, sin, norm=None):
            super().__init__()
            self.layers = nn.ModuleList(layers)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self.norm = norm

        def forward(self, hidden, causal_mask):
            position_embeddings = (self.rope_cos, self.rope_sin)
            for layer in self.layers:
                hidden = layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
            if self.norm is not None:
                hidden = self.norm(hidden)
            return hidden

    class ActionContext(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.state_fc1 = copy.deepcopy(head.state_encoder.fc1.linear)
            self.state_fc2 = copy.deepcopy(head.state_encoder.fc2.linear)
            self.num_heads = head.transformer_blocks[0].attn.num_heads
            embed_dim = head.embed_dim
            self.head_dim = embed_dim // self.num_heads
            self.k_weights = nn.ParameterList()
            self.k_biases = nn.ParameterList()
            self.v_weights = nn.ParameterList()
            self.v_biases = nn.ParameterList()
            for block in head.transformer_blocks:
                weight = block.attn.in_proj_weight.detach()
                bias = block.attn.in_proj_bias.detach()
                self.k_weights.append(nn.Parameter(weight[embed_dim : 2 * embed_dim].clone()))
                self.k_biases.append(nn.Parameter(bias[embed_dim : 2 * embed_dim].clone()))
                self.v_weights.append(nn.Parameter(weight[2 * embed_dim :].clone()))
                self.v_biases.append(nn.Parameter(bias[2 * embed_dim :].clone()))

        def forward(self, fused_tokens, context_mask, state):
            state_token = self.state_fc2(functional.relu(self.state_fc1(state))).unsqueeze(1)
            context = torch.cat((fused_tokens, state_token), dim=1)
            valid = torch.cat(
                (
                    context_mask.to(torch.bool),
                    torch.ones(
                        context_mask.shape[0],
                        1,
                        dtype=torch.bool,
                        device=context_mask.device,
                    ),
                ),
                dim=1,
            )
            key_mask = torch.where(
                valid[:, None, None, :],
                torch.zeros(1, dtype=context.dtype, device=context.device),
                torch.full(
                    (1,),
                    -10000.0,
                    dtype=context.dtype,
                    device=context.device,
                ),
            )
            outputs = [key_mask]
            batch, length, _ = context.shape
            for k_weight, k_bias, v_weight, v_bias in zip(
                self.k_weights,
                self.k_biases,
                self.v_weights,
                self.v_biases,
                strict=True,
            ):
                key = functional.linear(context, k_weight, k_bias)
                value = functional.linear(context, v_weight, v_bias)
                key = key.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
                value = value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
                outputs.extend((key, value))
            return tuple(outputs)

    class CachedActionBlock(nn.Module):
        def __init__(self, block):
            super().__init__()
            embed_dim = block.attn.embed_dim
            weight = block.attn.in_proj_weight.detach()
            bias = block.attn.in_proj_bias.detach()
            self.q_weight = nn.Parameter(weight[:embed_dim].clone())
            self.q_bias = nn.Parameter(bias[:embed_dim].clone())
            self.out_proj = copy.deepcopy(block.attn.out_proj)
            self.norm1 = copy.deepcopy(block.norm1)
            self.norm2 = copy.deepcopy(block.norm2)
            self.ff = copy.deepcopy(block.ff)
            self.num_heads = block.attn.num_heads
            self.head_dim = embed_dim // self.num_heads
            self.scale = self.head_dim**-0.5

        def forward(self, action_tokens, key, value, key_mask, time_embedding):
            batch, length, embed_dim = action_tokens.shape
            normalized = self.norm1(action_tokens)
            query = functional.linear(normalized, self.q_weight, self.q_bias)
            query = query.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
            weights = functional.softmax(scores + key_mask, dim=-1)
            attended = torch.matmul(weights, value)
            attended = attended.transpose(1, 2).contiguous().reshape(batch, length, embed_dim)
            hidden = action_tokens + self.out_proj(attended)
            ff_input = self.norm2(hidden) + time_embedding.unsqueeze(1)
            return hidden + self.ff(ff_input)

    class ActionStep(nn.Module):
        def __init__(self, head):
            super().__init__()
            encoder = head.action_encoder
            self.action_fc1 = copy.deepcopy(encoder.W1.linear)
            self.action_fc2 = copy.deepcopy(encoder.W2.linear)
            self.action_fc3 = copy.deepcopy(encoder.W3.linear)
            self.register_buffer(
                "action_position",
                encoder.pos_encoding.pe[:, : head.horizon].clone(),
            )
            self.register_buffer("time_position", head.time_pos_enc.pe.clone())
            self.blocks = nn.ModuleList(
                CachedActionBlock(block) for block in head.transformer_blocks
            )
            self.horizon = head.horizon

        def forward(self, action_seq, time_index, key_mask, *key_values):
            batch = action_seq.shape[0]
            hidden = action_seq.reshape(batch * self.horizon, action_seq.shape[-1])
            hidden = functional.relu(self.action_fc1(hidden))
            hidden = hidden.view(batch, self.horizon, -1) + self.action_position
            hidden = functional.relu(self.action_fc2(hidden.reshape(batch * self.horizon, -1)))
            hidden = self.action_fc3(hidden)
            hidden = hidden.view(batch, self.horizon, -1)
            time_embedding = torch.index_select(
                self.time_position.squeeze(0),
                0,
                time_index,
            )
            for index, block in enumerate(self.blocks):
                hidden = block(
                    hidden,
                    key_values[index * 2],
                    key_values[index * 2 + 1],
                    key_mask,
                    time_embedding,
                )
            return hidden

    class ActionOutput(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.norm = copy.deepcopy(head.norm_out)
            self.pool = copy.deepcopy(head.seq_pool_proj)
            self.fc1 = copy.deepcopy(head.mlp_head.fc1.linear)
            self.fc2 = copy.deepcopy(head.mlp_head.fc2.linear)
            self.horizon = head.horizon
            self.action_dim = head.per_action_dim

        def forward(self, hidden):
            hidden = self.norm(hidden)
            pooled = self.pool(hidden.reshape(hidden.shape[0], -1))
            velocity = self.fc2(functional.relu(self.fc1(pooled)))
            return velocity.view(hidden.shape[0], self.horizon, self.action_dim)

    return (
        VisionChunk,
        TokenEmbedding,
        LanguageChunk,
        ActionContext,
        ActionStep,
        ActionOutput,
    )


def _export_family(args: argparse.Namespace) -> None:
    import gc

    import torch

    policy = _build_policy(args.base, args.seed, args.seq_len,
                           args.max_views, args.checkpoint)
    model = policy.model
    (
        VisionChunk,
        TokenEmbedding,
        LanguageChunk,
        ActionContext,
        ActionStep,
        ActionOutput,
    ) = _build_wrappers()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict] = []

    def dump(module, sample, input_names, output_names, stem):
        path = args.out_dir / f"{stem}.onnx"
        temporary = args.out_dir / f".{stem}.onnx.tmp"
        module = module.eval().float()
        with torch.no_grad():
            torch.onnx.export(
                module,
                sample,
                str(temporary),
                input_names=input_names,
                output_names=output_names,
                opset_version=args.opset,
                dynamo=False,
                do_constant_folding=True,
            )
        temporary.replace(path)
        validate_onnx(path)
        record = {
            "name": stem,
            "file": path.name,
            "inputs": input_names,
            "outputs": output_names,
            "parameters": sum(parameter.numel() for parameter in module.parameters()),
            "state_elements": _state_elements(module),
            "size_mb": round(path.stat().st_size / 1e6, 1),
        }
        exported.append(record)
        print(
            f"  {stem:18s} {record['state_elements'] / 1e6:8.2f}M elements "
            f"{record['size_mb']:8.1f} MB",
            flush=True,
        )

    if args.family == "vision":
        owner = model.embedder.model
        tower = owner.vision_tower
        layers = list(tower.encoder.layer)
        if sum(VISION_SPLITS) != len(layers):
            raise ValueError("vision split does not match the model")
        hidden = torch.zeros(args.max_views, 3, 448, 448)
        start = 0
        for index, count in enumerate(VISION_SPLITS):
            last = index == len(VISION_SPLITS) - 1
            chunk = VisionChunk(
                copy.deepcopy(tower.embeddings) if index == 0 else None,
                [copy.deepcopy(layer) for layer in layers[start : start + count]],
                copy.deepcopy(owner.multi_modal_projector) if last else None,
            )
            start += count
            stem = f"vision_{index}"
            dump(
                chunk,
                (hidden,),
                ["pixel_values" if index == 0 else "hidden_in"],
                ["image_features" if last else "hidden_out"],
                stem,
            )
            with torch.no_grad():
                hidden = chunk(hidden)
            del chunk
            gc.collect()

    elif args.family == "language":
        language = model.embedder.model.language_model
        embedding = TokenEmbedding(copy.deepcopy(language.embed_tokens))
        ids = torch.zeros(1, args.seq_len, dtype=torch.long)
        dump(embedding, (ids,), ["input_ids"], ["token_embeddings"], "token_embedding")
        del embedding
        gc.collect()

        position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            cos, sin = language.rotary_emb(
                torch.zeros(1, args.seq_len, language.config.hidden_size),
                position_ids,
            )
        layers = list(language.layers)
        if sum(LANGUAGE_SPLITS) != len(layers):
            raise ValueError("language split does not match the truncated model")
        hidden = torch.zeros(1, args.seq_len, language.config.hidden_size)
        causal_mask = torch.zeros(1, 1, args.seq_len, args.seq_len)
        start = 0
        for index, count in enumerate(LANGUAGE_SPLITS):
            last = index == len(LANGUAGE_SPLITS) - 1
            chunk = LanguageChunk(
                [copy.deepcopy(layer) for layer in layers[start : start + count]],
                cos.clone(),
                sin.clone(),
                copy.deepcopy(language.norm) if last else None,
            )
            start += count
            stem = f"language_{index}"
            dump(
                chunk,
                (hidden, causal_mask),
                ["hidden_in", "causal_mask"],
                ["fused_tokens" if last else "hidden_out"],
                stem,
            )
            with torch.no_grad():
                hidden = chunk(hidden, causal_mask)
            del chunk
            gc.collect()

    elif args.family == "action":
        head = model.action_head
        context = ActionContext(head)
        fused = torch.zeros(1, args.seq_len, head.embed_dim)
        valid = torch.ones(1, args.seq_len, dtype=torch.bool)
        state = torch.zeros(1, 24)
        context_outputs = ["key_mask"]
        for index in range(len(head.transformer_blocks)):
            context_outputs.extend((f"key_{index}", f"value_{index}"))
        dump(
            context,
            (fused, valid, state),
            ["fused_tokens", "context_mask", "state"],
            context_outputs,
            "action_context",
        )
        with torch.no_grad():
            cached = context(fused, valid, state)

        step = ActionStep(head)
        action = torch.zeros(1, head.horizon, head.per_action_dim)
        time_index = torch.zeros(1, dtype=torch.long)
        step_inputs = ["action", "time_index", "key_mask"]
        for index in range(len(head.transformer_blocks)):
            step_inputs.extend((f"key_{index}", f"value_{index}"))
        dump(
            step,
            (action, time_index, *cached),
            step_inputs,
            ["action_hidden"],
            "action_step",
        )
        with torch.no_grad():
            action_hidden = step(action, time_index, *cached)

        output = ActionOutput(head)
        dump(
            output,
            (action_hidden,),
            ["action_hidden"],
            ["velocity"],
            "action_output",
        )
    else:
        raise ValueError(f"unsupported family {args.family}")

    identity = {
        "base_revision": BASE_REVISION,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "max_views": args.max_views,
        "opset": args.opset,
    }
    meta = {"family": args.family, "identity": identity, "graphs": exported}
    (args.out_dir / f"_meta_{args.family}.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )


def _ckpt_get(checkpoint: Path | None, key: str, default):
    """One field from a checkpoint's config.json, or the bootstrap default."""
    if checkpoint is None:
        return default
    path = checkpoint / "config.json"
    if not path.is_file():
        return default
    value = json.loads(path.read_text()).get(key)
    return default if value is None else value


def _checkpoint_provenance(checkpoint: Path | None) -> dict | None:
    """Identify the trained weights a bundle was built from.

    Records the weight file's own sha256, not just the directory name: a bundle that
    cannot say which weights produced it cannot be told apart from one built by an
    interrupted or re-pointed export.
    """
    if checkpoint is None:
        return None
    weights = checkpoint / "model.safetensors"
    digest = None
    if weights.is_file():
        h = hashlib.sha256()
        with weights.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        digest = h.hexdigest()
    config = {}
    config_path = checkpoint / "config.json"
    if config_path.is_file():
        raw = json.loads(config_path.read_text())
        config = {k: raw.get(k) for k in
                  ("chunk_size", "num_inference_timesteps", "max_state_dim",
                   "max_action_dim", "max_views", "vlm_model_name")}
    return {
        "path": str(checkpoint),
        "model_safetensors_sha256": digest,
        "model_safetensors_bytes": weights.stat().st_size if weights.is_file() else None,
        "config": config,
    }


def _copy_tokenizer(base: Path, out: Path) -> None:
    target = out / "tokenizer"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    names = {
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    }
    for path in base.iterdir():
        if path.is_file() and (
            path.name.startswith("tokenizer") or path.name in names
        ):
            shutil.copy2(path, target / path.name)




def _parent(args: argparse.Namespace) -> None:
    revision = (args.base / "REVISION").read_text().strip()
    if revision != BASE_REVISION:
        raise SystemExit(
            f"base revision {revision} does not match pinned {BASE_REVISION}"
        )
    # Every view costs image_seq_length positions whether or not a camera fills it --
    # the embedder pads the stack to max_views and masks the absent ones, so the budget
    # is spent either way. It raises rather than silently truncating the prompt, so
    # check here where the message can name the fix.
    floor = args.max_views * IMAGE_SEQ_LENGTH + 8
    if args.seq_len < floor:
        raise SystemExit(
            f"seq_len {args.seq_len} is too short for {args.max_views} view(s): each "
            f"needs {IMAGE_SEQ_LENGTH} positions, leaving nothing for text. "
            f"Use --seq-len {floor} or more.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "base_revision": BASE_REVISION,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "max_views": args.max_views,
        "opset": args.opset,
    }

    for family in () if args.bundle_only else args.families:
        command = [
            sys.executable,
            __file__,
            "--_family",
            family,
            "--base",
            str(args.base),
            "--out-dir",
            str(args.out_dir),
            "--seed",
            str(args.seed),
            "--seq-len",
            str(args.seq_len),
            "--max-views",
            str(args.max_views),
            "--opset",
            str(args.opset),
        ] + (["--checkpoint", str(args.checkpoint)] if args.checkpoint else [])
        print(f"[{family}]", flush=True)
        if subprocess.run(command).returncode != 0:
            raise SystemExit(f"{family} export failed")

    metas = []
    for family in FAMILIES:
        path = args.out_dir / f"_meta_{family}.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}; export every family before bundling")
        document = json.loads(path.read_text())
        if document.get("identity") != identity:
            raise SystemExit(f"{path} belongs to a different export contract")
        metas.append(document)

    graphs = [
        graph
        for document in metas
        for graph in document["graphs"]
    ]
    _copy_tokenizer(args.base, args.out_dir)
    bundle = {
        "schema_version": 1,
        "model": "evo1",
        # A trained checkpoint flips all three. The bootstrap's warning is not
        # boilerplate -- its action head is random -- so it must not survive onto a
        # bundle that has real weights, and must not be dropped from one that does not.
        "deployable": bool(args.checkpoint),
        "random_action_head": not args.checkpoint,
        "warning": None if args.checkpoint else (
            "Infrastructure-validation bundle only. The action head is deterministic "
            "random initialization and must never control a robot."
        ),
        "base": {"repo_id": MODEL_ID, "revision": BASE_REVISION},
        "checkpoint": _checkpoint_provenance(args.checkpoint),
        "seed": args.seed,
        "max_views": args.max_views,
        "valid_views": args.max_views,
        "image_size": 448,
        "image_seq_length": 256,
        "image_token_id": 151667,
        "seq_len": args.seq_len,
        "hidden_size": 896,
        "vision_hidden_size": 1024,
        "vision_splits": list(VISION_SPLITS),
        "language_splits": list(LANGUAGE_SPLITS),
        # Bootstrap defaults, overridden below by a checkpoint's own config. A bundle
        # that hardcoded these would run a different policy than the weights encode
        # the moment someone exports a checkpoint shaped differently, and nothing
        # downstream would notice.
        "chunk_size": _ckpt_get(args.checkpoint, "chunk_size", 50),
        "max_state_dim": _ckpt_get(args.checkpoint, "max_state_dim", 24),
        "max_action_dim": _ckpt_get(args.checkpoint, "max_action_dim", 24),
        "num_inference_timesteps": _ckpt_get(
            args.checkpoint, "num_inference_timesteps", 32),
        "graphs": graphs,
        "tokenizer": {"path": "tokenizer", "padding_side": "right"},
        "fixture": None,
        "provenance": {
            "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "lerobot": package_version("lerobot"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "onnx": package_version("onnx"),
            "opset": args.opset,
        },
    }
    (args.out_dir / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    manifest_files = write_manifest(args.out_dir)
    total = sum(graph["state_elements"] for graph in graphs)
    largest = max(graph["state_elements"] for graph in graphs)
    print(f"\n{len(graphs)} graphs, {total / 1e6:.2f}M serialized elements")
    print(f"largest graph: {largest / 1e6:.2f}M elements")
    print(f"wrote {args.out_dir / 'bundle.json'}")
    print(f"wrote MANIFEST.sha256 ({manifest_files} files)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "models" / "InternVL3-1B-hf",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "exports" / "split-bootstrap",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=320)
    parser.add_argument("--views", "--max-views", type=int, default=1, dest="max_views",
                        help="how many camera views the bundle is sized for. (--max-views is "
                             "the old name and still works.) Fixes the vision graph's static "
                             "view count; the runtime zero-pads any it does not have.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="a trained LeRobot Evo1 policy directory to export instead of a fresh "
             "bootstrap head (e.g. zuoxingdong/evo1_libero downloaded locally). "
             "Without it the export is the nondeployable random-head bootstrap.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
    )
    parser.add_argument("--bundle-only", action="store_true")
    parser.add_argument("--_family", dest="family", choices=FAMILIES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base = args.base.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.family:
        _export_family(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
