#!/usr/bin/env python3
# Vendored from the author's fine-tuning pipeline: vla-onnx/xvla/export_split_onnx.py @ 3f7793d (robot-specific comments trimmed).
"""Export X-VLA as split ONNX graphs sized for the 8 GB Orin.

See ../notes/split_design.md for the measured build-memory curve that sets the cut sizes.
Short version: TRT's build peak is ~3.18 GB + 5.63 x (FP32 weight GB), so an engine must
carry at most ~0.40 GB of weights (~100 M params) to build with real margin. X-VLA's three
heavy components are all over that on their own, so each is split by parameter budget:

    vision (1.44 GB)  DaViT convs/blocks, flattened and packed; last chunk adds the projector
    text   (0.83 GB)  BART embedding + 12 encoder layers
    denoise (1.21 GB) 24 transformer blocks, run once per denoising step

`generate_actions` runs the VLM once and the denoiser `num_denoising_steps` times, so the
loop-invariant conditioning projections are hoisted into their own `cond` graph -- exact,
not an approximation, because the sequence positions are fixed.

Each graph is exported in its own subprocess: loading the policy costs ~3.5 GB on CPU and
holding that alongside an export trace is what pushes this board into swap.

    python tools/export_split_onnx.py --checkpoint models/xvla-base --domain-id 0
    python tools/export_split_onnx.py --random-init --graphs vision   # smoke-test only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from vla_common.bundle import validate_onnx, write_manifest
from bundle_contract import (SCHEMA_VERSION, build_processor_contract,
                                          copy_processor_artifacts,
                                          materialize_tokenizer, tree_sha256)

GRAPHS = ("vision", "text_encoder", "cond", "denoise")

# Per-engine FP32 weight budget, from tools/build_probe.py on this board. 0.40 GB built
# with 1.69 GB of headroom; 0.60 GB finished with only 0.47 GB left.
DEFAULT_BUDGET_GB = 0.40


def _vlm_parts(vlm):
    """(owner, vision_tower, multi_modal_projector, language_model) across transformers versions.

    transformers < 5 hangs the vision tower, the projector and the language model
    directly off `Florence2ForConditionalGeneration`. transformers 5.x moved them one
    level down onto a `Florence2Model` at `.model`, and only `get_image_features`
    survives as a forwarding method — so `vlm.multi_modal_projector` raises
    AttributeError there. Resolve through whichever level actually owns the modules,
    and return that owner so `del owner.vision_tower` frees the right reference.
    """
    owner = getattr(vlm, "model", None)
    if owner is None or not hasattr(owner, "language_model"):
        owner = vlm
    # getattr with a default, not attribute access: the text_encoder branch DELETES
    # vision_tower to free 1.4 GB before tracing, and then builds TextHead, which calls
    # back in here. Requiring the tower would make the second call fail on the first
    # call's cleanup.
    return (owner,
            getattr(owner, "vision_tower", None),
            getattr(owner, "multi_modal_projector", None),
            owner.language_model)


def _n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())



def _plan_chunks(units: list[tuple[str, object]], budget_params: int) -> list[list[int]]:
    """Greedy contiguous packing of (name, module) units into chunks under the budget.

    Contiguous because these are sequential stacks: a chunk must be a run of adjacent
    layers for the intermediate tensor to be the only thing crossing the boundary.
    """
    chunks: list[list[int]] = []
    current: list[int] = []
    running = 0
    for i, (_, mod) in enumerate(units):
        n = _n_params(mod)
        if current and running + n > budget_params:
            chunks.append(current)
            current, running = [], 0
        current.append(i)
        running += n
    if current:
        chunks.append(current)
    return chunks


# ======================================================================================
# graph wrappers
# ======================================================================================


def _build_wrappers():
    """Imported lazily inside the child process so the parent never loads torch."""
    import torch
    from torch import nn

    from lerobot.policies.xvla.soft_transformer import timestep_embedding

    class SeqChunk(nn.Module):
        """A run of DaViT convs/blocks; the final chunk also applies the projector."""

        def __init__(self, mods, projector=None):
            super().__init__()
            self.mods = nn.ModuleList(mods)
            self.projector = projector

        def forward(self, x):
            for m in self.mods:
                x = m(x)
            if self.projector is not None:
                x = self.projector(x)
            return x

    class TextHead(nn.Module):
        """Token embedding + BART preamble + the first run of encoder layers.

        `BartEncoder.forward` cannot be reused here: it unconditionally calls
        `create_bidirectional_mask`, which does not survive JIT tracing (it reaches
        `sdpa_mask` with a tuple where it expects a tensor, raising IndexError), and
        passing `attention_mask=None` does not skip that call. So the preamble is
        reproduced explicitly -- it is exactly the four lines below, taken from
        `BartEncoder.forward`, minus the mask construction and the eval-time no-op
        dropout.

        Dropping the mask is not a simplification of the model: `forward_vlm` builds an
        all-ones mask over image+language tokens (it does not mask language padding), so
        every token attends to every token, which is what `attn_mask=None` means.
        `parity.py` checks that equivalence numerically against the reference.
        """

        def __init__(self, vlm, keep_layers: int):
            super().__init__()
            self.embed = vlm.get_input_embeddings()
            encoder = _vlm_parts(vlm)[3].encoder
            self.embed_positions = encoder.embed_positions
            self.layernorm_embedding = encoder.layernorm_embedding
            self.layers = nn.ModuleList(list(encoder.layers[:keep_layers]))

        def forward(self, input_ids, image_tokens):
            inputs_embeds = self.embed(input_ids)
            merged = torch.cat([image_tokens, inputs_embeds], dim=1)
            hidden = merged + self.embed_positions(merged[:, :, -1])
            hidden = self.layernorm_embedding(hidden)
            for layer in self.layers:
                hidden = layer(hidden, None)
            return hidden

    class TextChunk(nn.Module):
        """Remaining BART encoder layers.

        `attention_mask=None` is correct here, not a shortcut: `forward_vlm` builds an
        all-ones mask over image+language tokens (it does not mask language padding), so
        every token attends to every token.
        """

        def __init__(self, layers):
            super().__init__()
            self.layers = nn.ModuleList(layers)

        def forward(self, hidden):
            for layer in self.layers:
                hidden = layer(hidden, None)
            return hidden

    class CondGraph(nn.Module):
        """The hoisted loop-invariant half of `SoftPromptedTransformer.forward`.

        vlm_features [1,T_v,D], aux_visual [1,T_a,D] -> cond_tokens [1, T_v+T_a, H],
        already carrying their `pos_emb` slice. Positions are fixed, so folding the slice
        in here is equivalent to adding it inside the denoising loop.
        """

        def __init__(self, transformer, n_action: int):
            super().__init__()
            self.vlm_proj = transformer.vlm_proj
            self.aux_visual_proj = transformer.aux_visual_proj
            self.register_buffer("pos_emb", transformer.pos_emb.detach().clone())
            self.n_action = n_action

        def forward(self, vlm_features, aux_visual):
            cond = torch.cat(
                [self.vlm_proj(vlm_features), self.aux_visual_proj(aux_visual)], dim=1
            )
            return cond + self.pos_emb[:, self.n_action : self.n_action + cond.shape[1]]

    class DenoiseGraph(nn.Module):
        """One slice of the hot path, run once per denoising step.

        `first` slices normally take (x_t, t, proprio, cond_tokens) and assemble the
        sequence. With `fuse_interpolation`, they instead take
        (x1, action, t, proprio, cond_tokens) and form the exact X-VLA interpolation on
        device;
        `last` slices apply the final norm + action decoder. `domain_id` is baked: the
        DomainAwareLinear weights and the soft prompts are gathered at export time and
        become constants, so the 30-domain tables never enter an engine.
        """

        def __init__(self, transformer, action_space, blocks, *, first, last, domain_id,
                     dim_time, n_action, dim_action=None, dim_proprio=None,
                     fuse_interpolation=False):
            super().__init__()
            self.blocks = blocks
            self.first = first
            self.last = last
            self.dim_time = dim_time
            self.n_action = n_action
            self.fuse_interpolation = bool(fuse_interpolation)

            did = torch.tensor([domain_id], dtype=torch.long)
            if first:
                # EE6DActionSpace.preprocess zeroes the gripper channels. Done with
                # constant masks built here, in eager mode, rather than `ones_like` plus
                # an indexed assignment inside forward: the TorchScript exporter mistraced
                # that into Add/Expand nodes with empty input names, producing an ONNX
                # file that loads nowhere ("input 0 is marked single but has an empty
                # string"). Shapes are static, so a constant mask is equivalent.
                # AutoActionSpace has no grippers: it only pads real axes to the model
                # width and trims them after inference, so the correct mask is all ones.
                gripper_idx = list(getattr(action_space, "gripper_idx", ()))
                amask = torch.ones(1, 1, dim_action)
                amask[..., gripper_idx] = 0.0
                self.register_buffer("action_mask", amask)
                pmask = torch.ones(1, dim_proprio)
                pmask[..., gripper_idx] = 0.0
                self.register_buffer("proprio_mask", pmask)

                enc = transformer.action_encoder
                self.register_buffer(
                    "enc_w", enc.fc(did).view(enc.input_size, enc.output_size).detach().clone()
                )
                self.register_buffer("enc_b", enc.bias(did).view(1, 1, -1).detach().clone())
                self.register_buffer(
                    "pos_emb_action", transformer.pos_emb[:, :n_action].detach().clone()
                )
                self.register_buffer(
                    "soft_prompts",
                    transformer.soft_prompt_hub(did)
                    .view(1, transformer.len_soft_prompts, transformer.hidden_size)
                    .detach()
                    .clone(),
                )
            if last:
                dec = transformer.action_decoder
                self.norm = transformer.norm
                self.register_buffer(
                    "dec_w", dec.fc(did).view(dec.input_size, dec.output_size).detach().clone()
                )
                self.register_buffer("dec_b", dec.bias(did).view(1, 1, -1).detach().clone())

        def forward(self, *args):
            if self.first:
                if self.fuse_interpolation:
                    x1, action, t, proprio, cond_tokens = args
                    weight = t.view(-1, 1, 1)
                    x_t = x1 * weight + action * (1.0 - weight)
                else:
                    x_t, t, proprio, cond_tokens = args
                x_t = x_t * self.action_mask
                proprio = proprio * self.proprio_mask

                # `repeat` with static counts, not `expand(-1, n, -1)`: the -1s traced
                # into Expand nodes with empty inputs (see the mask comment above).
                time_emb = timestep_embedding(t, self.dim_time)
                time_tokens = time_emb.unsqueeze(1).repeat(1, self.n_action, 1)
                proprio_tokens = proprio.unsqueeze(1).repeat(1, self.n_action, 1)
                tokens = torch.cat([x_t, proprio_tokens, time_tokens], dim=-1)

                x = torch.matmul(tokens, self.enc_w) + self.enc_b
                x = x + self.pos_emb_action
                x = torch.cat([x, cond_tokens, self.soft_prompts], dim=1)
            else:
                (x,) = args

            for block in self.blocks:
                x = block(x)

            if not self.last:
                return x
            head = self.norm(x[:, : self.n_action])
            return torch.matmul(head, self.dec_w) + self.dec_b

    return SeqChunk, TextHead, TextChunk, CondGraph, DenoiseGraph


# ======================================================================================
# child: export one graph family
# ======================================================================================


def export_one(args) -> None:
    import gc

    import torch
    from torch import nn

    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from lerobot.configs.policies import PreTrainedConfig

    # The checkpoint's config says device=cuda, and both constructors honour it. Export
    # on CPU: the trace samples are CPU tensors, and 3.5 GB of weights on an Orin's
    # unified memory would come straight out of the same budget the build needs.
    cfg_obj = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    cfg_obj.device = "cpu"

    if args.random_init:
        # Exercises the export path (tracing, opset coverage, shapes) without the 3.5 GB
        # checkpoint. Topology and shapes are identical; only values differ, so this
        # catches "DaViT will not trace" long before the weights land.
        policy = XVLAPolicy(cfg_obj)
    else:
        policy = XVLAPolicy.from_pretrained(str(args.checkpoint), config=cfg_obj)
    policy.to("cpu").eval()
    model = policy.model
    cfg = policy.config

    SeqChunk, TextHead, TextChunk, CondGraph, DenoiseGraph = _build_wrappers()

    hidden = cfg.hidden_size
    n_views = cfg.num_image_views
    img_hw = tuple(cfg.resize_imgs_with_padding or (224, 224))
    lang_len = args.lang_len
    proj_dim = model.vlm.config.vision_config.projection_dim
    budget = int(args.budget_gb * 1e9 / 4)

    meta: dict = {}
    exported: list[dict] = []

    def dump(module, sample, names_in, names_out, stem):
        path = out_dir / f"{stem}.onnx"
        module = module.eval().float()
        with torch.no_grad():
            torch.onnx.export(
                module, sample, str(path),
                input_names=names_in, output_names=names_out,
                opset_version=args.opset, dynamo=False,
            )
        validate_onnx(path, stem)
        exported.append({
            "name": stem,
            "file": path.name,
            "inputs": names_in,
            "outputs": names_out,
            "params": _n_params(module),
            "size_mb": round(path.stat().st_size / 1e6, 1),
        })
        print(f"  {stem}: {_n_params(module) / 1e6:.1f}M params, "
              f"{path.stat().st_size / 1e6:.0f} MB", flush=True)

    # -- vision ------------------------------------------------------------------------
    if args.graph == "vision":
        _, tower, projector, _ = _vlm_parts(model.vlm)
        units: list[tuple[str, object]] = []
        for si, (conv, block) in enumerate(zip(tower.convs, tower.blocks)):
            units.append((f"conv{si}", conv))
            for li, layer in enumerate(block):
                units.append((f"stage{si}.layer{li}", layer))

        plan = _plan_chunks(units, budget)
        print(f"  vision: {len(units)} units -> {len(plan)} engines "
              f"(budget {args.budget_gb:.2f} GB)")

        x = torch.zeros(args.valid_views, 3, *img_hw)
        with torch.no_grad():
            for ci, idxs in enumerate(plan):
                last = ci == len(plan) - 1
                chunk = SeqChunk(
                    [units[i][1] for i in idxs], projector=projector if last else None
                )
                stem = "vision" if len(plan) == 1 else f"vision_{ci}"
                name_in = "pixel_values" if ci == 0 else "hidden_in"
                name_out = "image_features" if last else "hidden_out"
                dump(chunk, (x,), [name_in], [name_out], stem)
                x = chunk(x)
        meta["tokens_per_view"] = int(x.shape[1])

    # -- text encoder ------------------------------------------------------------------
    elif args.graph == "text_encoder":
        _owner, _, _, _lm = _vlm_parts(model.vlm)
        del _owner.vision_tower
        gc.collect()
        vlm = model.vlm
        layers = list(_lm.encoder.layers)
        embed_params = _n_params(vlm.get_input_embeddings())

        # The embedding table rides with the head chunk, so it eats into that budget.
        units = [(f"layer{i}", layer) for i, layer in enumerate(layers)]
        head_budget = max(budget - embed_params, _n_params(layers[0]))
        first = _plan_chunks(units, head_budget)[0]
        rest = _plan_chunks(units[len(first):], budget)
        print(f"  text: embed {embed_params / 1e6:.1f}M + {len(layers)} layers -> "
              f"{1 + len(rest)} engines")

        ids = torch.zeros(1, lang_len, dtype=torch.long)
        img = torch.zeros(1, args.tokens_per_view, proj_dim)
        head = TextHead(vlm, keep_layers=len(first))
        stem = "text_encoder" if not rest else "text_encoder_0"
        dump(head, (ids, img), ["input_ids", "image_tokens"],
             ["vlm_features" if not rest else "hidden_out"], stem)

        with torch.no_grad():
            h = head(ids, img)
        offset = len(first)
        for ci, idxs in enumerate(rest):
            chunk = TextChunk([layers[offset + i] for i in idxs])
            last = ci == len(rest) - 1
            dump(chunk, (h,), ["hidden_in"],
                 ["vlm_features" if last else "hidden_out"], f"text_encoder_{ci + 1}")
            with torch.no_grad():
                h = chunk(h)

    # -- conditioning ------------------------------------------------------------------
    elif args.graph == "cond":
        cg = CondGraph(model.transformer, cfg.chunk_size)
        del model.vlm
        gc.collect()
        t_v = args.tokens_per_view + lang_len
        t_a = (n_views - 1) * args.tokens_per_view
        dump(cg, (torch.zeros(1, t_v, proj_dim), torch.zeros(1, t_a, proj_dim)),
             ["vlm_features", "aux_visual"], ["cond_tokens"], "cond")

    # -- denoiser ----------------------------------------------------------------------
    elif args.graph == "denoise":
        del model.vlm
        gc.collect()
        tr = model.transformer
        splits = args.policy_splits
        if sum(splits) != cfg.depth:
            sys.exit(f"--policy-splits {splits} must sum to depth {cfg.depth}")

        t_v = args.tokens_per_view + lang_len
        t_a = (n_views - 1) * args.tokens_per_view
        n_cond = t_v + t_a
        seq_len = cfg.chunk_size + n_cond + cfg.len_soft_prompts
        meta["seq_len"] = seq_len
        print(f"  denoise: {cfg.depth} blocks -> {len(splits)} engines {splits}, "
              f"seq_len {seq_len}")

        start = 0
        for i, count in enumerate(splits):
            blocks = nn.ModuleList(list(tr.blocks[start : start + count]))
            start += count
            first_c, last_c = i == 0, i == len(splits) - 1
            dg = DenoiseGraph(
                tr, model.action_space, blocks,
                first=first_c, last=last_c, domain_id=args.domain_id,
                dim_time=cfg.dim_time, n_action=cfg.chunk_size,
                dim_action=model.dim_action, dim_proprio=model.dim_proprio,
                fuse_interpolation=args.fuse_denoise_interpolation,
            )
            stem = "denoise" if len(splits) == 1 else f"denoise_{i}"
            if first_c:
                action_sample = torch.zeros(1, cfg.chunk_size, model.dim_action)
                common_sample = (
                    torch.zeros(1), torch.zeros(1, model.dim_proprio),
                    torch.zeros(1, n_cond, hidden),
                )
                if args.fuse_denoise_interpolation:
                    sample = (action_sample, action_sample, *common_sample)
                    names_in = ["x1", "action", "t", "proprio", "cond_tokens"]
                else:
                    sample = (action_sample, *common_sample)
                    names_in = ["x_t", "t", "proprio", "cond_tokens"]
            else:
                sample = (torch.zeros(1, seq_len, hidden),)
                names_in = ["hidden_in"]
            dump(dg, sample, names_in, ["action"] if last_c else ["hidden_out"], stem)
        meta["denoise_input_mode"] = (
            "fused_interpolation" if args.fuse_denoise_interpolation else "x_t")

    graph_identity = {
        "checkpoint_tree_sha256": args.checkpoint_sha,
        "random_init": bool(args.random_init),
        "domain_id": args.domain_id,
        "valid_views": args.valid_views,
        "lang_len": args.lang_len,
        "num_image_views": n_views,
        "chunk_size": cfg.chunk_size,
        "max_state_dim": model.dim_proprio,
        "max_action_dim": model.dim_action,
        "opset": args.opset,
        "budget_gb": args.budget_gb,
        "policy_splits": args.policy_splits,
    }
    (out_dir / f"_meta_{args.graph}.json").write_text(
        json.dumps({"graphs": exported, "identity": graph_identity, **meta}, indent=2)
    )


# ======================================================================================
# parent: orchestrate one subprocess per graph family
# ======================================================================================


def _dataset_fps(checkpoint: Path) -> int | None:
    """Playback rate of the dataset this checkpoint was fine-tuned on.

    The action chunk is RATE commands sampled at the training rate, so playing it at a
    different rate scales every motion the machine makes, with nothing downstream able to
    notice. `xvla_split.resolve_fps` therefore refuses to start without one rather than
    assuming 30 the way the SmolVLA path can.

    Unlike the task strings this needs no parquet reader -- meta/info.json is plain JSON --
    so it resolves even from the export venv, which carries neither pandas nor pyarrow.
    """
    cfg_f = checkpoint / "train_config.json"
    if not cfg_f.is_file():
        return None
    try:
        root = json.loads(cfg_f.read_text()).get("dataset", {}).get("root")
        if not root:
            return None
        info = Path(root) / "meta" / "info.json"
        if not info.is_file():
            return None
        return json.loads(info.read_text()).get("fps")
    except Exception as exc:
        print(f"!! cannot read the training fps from {cfg_f}: {type(exc).__name__}: {exc}")
        return None


def _dataset_tasks(checkpoint: Path) -> list[str]:
    """The instruction strings this checkpoint was fine-tuned against, in task_index order.

    The runtime has no default for these: `xvla_split.resolve_tasks` takes them from the
    bundle, or from --task, or it refuses to start -- because the policy conditions on the
    language embedding and a phrasing it never trained on is out of distribution with
    nothing downstream able to notice. Leaving them out therefore does not produce a
    bundle that merely lacks a nicety; it produces one that cannot be launched without
    the operator retyping the strings by hand, exactly right, from memory.

    A multi-task fine-tune records the whole LIST, in dataset order. `train_config.json` records the dataset
    root as an absolute path, so this works for a local fine-tune and returns [] for a base
    checkpoint that was never trained here -- which is correct: xvla-base has no task.
    """
    cfg_f = checkpoint / "train_config.json"
    if not cfg_f.is_file():
        return []
    try:
        root = json.loads(cfg_f.read_text()).get("dataset", {}).get("root")
    except Exception:
        return []
    if not root:
        return []
    tasks_f = Path(root) / "meta" / "tasks.parquet"
    if not tasks_f.is_file():
        return []
    try:
        import pandas as pd
        col = pd.read_parquet(tasks_f)
        # task_index is the integer the dataset's own task_index column points at, and
        # the order the D-pad cycles in. LeRobot v3 puts the string in the INDEX and
        # task_index in a column; older writers did the reverse, so handle both.
        if "task_index" in col.columns:
            col = col.sort_values("task_index")
        return ([str(t) for t in col["task"]] if "task" in col.columns
                else [str(i) for i in col.index])
    except Exception as exc:
        # LOUD, not silent. This export venv is deliberately a different one from the
        # training venv (lerobot 0.6.1 vs 0.5.1) and carries neither pandas nor pyarrow,
        # so it cannot read tasks.parquet at all -- a `return []` here is indistinguishable
        # from "this is a base checkpoint with no task", and quietly ships a bundle the
        # runtime will refuse to launch. Resolve them in the training venv and pass --task.
        print(f"!! cannot read {tasks_f}: {type(exc).__name__}: {exc}\n"
              f"   pass the instruction(s) explicitly with --task instead.")
        return []


def _provenance(checkpoint: Path, opset: int) -> dict:
    """Who built this bundle, from what, with which library versions.

    X-VLA's split is budget-driven: change the budget, the transformers version or the
    checkpoint and you get a different set of graphs that still loads. Without this
    block a bundle cannot be told apart from another bundle after the fact, which
    matters more here than on the SmolVLA side because this one is meant to be
    published and re-downloaded by people who did not build it.
    """
    import subprocess

    def _ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    try:
        sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                              "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        sha = None

    return {
        "checkpoint": str(checkpoint),
        "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "exporter_sha": sha,
        "opset": opset,
        "lerobot": _ver("lerobot"),
        "torch": _ver("torch"),
        "transformers": _ver("transformers"),
    }



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=REPO / "models" / "xvla-base")
    ap.add_argument("--out-dir", type=Path, default=REPO / "exports" / "split")
    ap.add_argument("--domain-id", type=int, default=0,
                    help="baked into the action encoder/decoder and soft prompts")
    ap.add_argument("--views", "--valid-views", type=int, default=None, dest="valid_views",
                    help="how many camera views the bundle is sized for. (--valid-views is "
                         "the old name and still works.) Sets the static batch of the vision "
                         "engine; defaults to the checkpoint's REAL camera count (its VISUAL "
                         "input_features), falling back to num_image_views when those are empty. "
                         "Padded views are zeroed by the runtime and never need a forward pass.")
    ap.add_argument("--lang-len", type=int, default=None,
                    help="language tokens; defaults to the saved policy preprocessor")
    ap.add_argument("--fps", type=int, default=None,
                    help="control rate the actions were authored at. Auto-resolved from "
                         "the checkpoint's training dataset; the runtime REFUSES to start "
                         "without one, because the chunk is rate commands and a wrong rate "
                         "silently scales every motion.")
    ap.add_argument("--task", action="append", dest="tasks", default=None,
                    help="instruction string, letter for letter as the training dataset "
                         "records it. Auto-resolved from the checkpoint's dataset, "
                         "including every task of a multi-task one. Repeatable: pass once "
                         "per instruction to override, first one first (the order the "
                         "runtime's D-pad cycles in).")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer path/id; defaults to the saved policy preprocessor")
    ap.add_argument("--budget-gb", type=float, default=DEFAULT_BUDGET_GB,
                    help="FP32 weight budget per engine (see notes/split_design.md)")
    ap.add_argument("--policy-splits", type=int, nargs="+", default=[6, 6, 6, 6],
                    help="block counts per denoise engine, must sum to depth (24)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--graphs", nargs="+", default=list(GRAPHS), choices=GRAPHS)
    ap.add_argument("--random-init", action="store_true",
                    help="build from config with random weights instead of loading the "
                         "checkpoint -- smoke-tests the export path only")
    ap.add_argument("--bundle-only", action="store_true",
                    help="rebuild bundle.json from the _meta_*.json already in --out-dir, "
                         "without exporting anything")
    ap.add_argument("--fuse-denoise-interpolation", action="store_true",
                    help="make denoise_0 consume x1 + previous action and form x_t "
                         "inside the graph for a fully device-resident denoise loop")
    ap.add_argument("--_graph", dest="graph", help=argparse.SUPPRESS)
    ap.add_argument("--tokens-per-view", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--checkpoint-sha", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.graph:  # child invocation
        if not args.checkpoint_sha:
            sys.exit("internal --checkpoint-sha is required for graph export")
        export_one(args)
        return

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    processor_contract = build_processor_contract(args.checkpoint, cfg)
    processor_lang_len = int(processor_contract["tokenizer"]["max_length"])
    if args.lang_len is None:
        args.lang_len = processor_lang_len
    elif args.lang_len != processor_lang_len:
        sys.exit(
            f"--lang-len {args.lang_len} conflicts with the checkpoint processor's "
            f"max_length={processor_lang_len}")
    tokenizer_source = args.tokenizer or processor_contract["tokenizer"]["source"]
    if not tokenizer_source:
        sys.exit("checkpoint processor has no tokenizer_name; pass --tokenizer")
    checkpoint_tree_sha = tree_sha256(args.checkpoint)

    if args.valid_views is None:
        # The camera count is the checkpoint's VISUAL input_features, NOT num_image_views:
        # that is the padded slot count (3 in xvla-base and in every finetune of it). The
        # old default baked 3 views into a 1-camera fine-tune's bundle, which the runtime
        # refuses ("bundle requires exactly 3 camera views"). xvla-base itself declares 3
        # VISUAL features, so the base export is unchanged.
        cams = [k for k, v in (cfg.get("input_features") or {}).items()
                if v.get("type") == "VISUAL"]
        args.valid_views = len(cams) or cfg.get("num_image_views") or 3
    num_views = int(cfg.get("num_image_views") or 0)
    if args.valid_views <= 0 or args.valid_views > num_views:
        sys.exit(
            f"--valid-views must be in [1, {num_views}], got {args.valid_views}")
    if processor_contract["action_mode"] != cfg.get("action_mode"):
        sys.exit("processor action mode does not match config.json")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"exporting X-VLA split graphs -> {args.out_dir}")
    print(f"  domain_id={args.domain_id}  valid_views={args.valid_views}  "
          f"lang_len={args.lang_len}  budget={args.budget_gb:.2f} GB  "
          f"policy_splits={args.policy_splits}\n")

    tokens_per_view = args.tokens_per_view
    for graph in [] if args.bundle_only else args.graphs:
        if graph != "vision" and tokens_per_view is None:
            sys.exit(f"'{graph}' needs tokens_per_view: export 'vision' first, or pass "
                     f"--tokens-per-view")
        print(f"[{graph}]", flush=True)
        cmd = [
            sys.executable, __file__, "--_graph", graph,
            "--checkpoint", str(args.checkpoint), "--out-dir", str(args.out_dir),
            "--domain-id", str(args.domain_id), "--valid-views", str(args.valid_views),
            "--lang-len", str(args.lang_len), "--opset", str(args.opset),
            "--checkpoint-sha", checkpoint_tree_sha,
            "--budget-gb", str(args.budget_gb),
            "--policy-splits", *[str(s) for s in args.policy_splits],
        ]
        if tokens_per_view is not None:
            cmd += ["--tokens-per-view", str(tokens_per_view)]
        if args.random_init:
            cmd.append("--random-init")
        if args.fuse_denoise_interpolation:
            cmd.append("--fuse-denoise-interpolation")
        if subprocess.run(cmd, text=True).returncode != 0:
            sys.exit(f"export of '{graph}' failed")

        meta = json.loads((args.out_dir / f"_meta_{graph}.json").read_text())
        if "tokens_per_view" in meta:
            tokens_per_view = meta["tokens_per_view"]
            print(f"  tokens_per_view = {tokens_per_view}")

    # Collect from every _meta_*.json on disk, not just this run's --graphs, so a partial
    # re-export (e.g. redoing text_encoder alone) still writes a complete bundle.
    order = {name: i for i, name in enumerate(GRAPHS)}
    metas = sorted(
        args.out_dir.glob("_meta_*.json"),
        key=lambda p: order.get(p.stem[len("_meta_"):], len(order)),
    )
    expected_graph_identity = {
        "checkpoint_tree_sha256": checkpoint_tree_sha,
        "random_init": bool(args.random_init),
        "domain_id": args.domain_id,
        "valid_views": args.valid_views,
        "lang_len": args.lang_len,
        "num_image_views": num_views,
        "chunk_size": cfg.get("chunk_size"),
        "max_state_dim": processor_contract["state"]["model_dim"],
        "max_action_dim": processor_contract["action"]["model_dim"],
        "opset": args.opset,
        "budget_gb": args.budget_gb,
        "policy_splits": args.policy_splits,
    }
    meta_docs = [json.loads(path.read_text()) for path in metas]
    for path, meta_doc in zip(metas, meta_docs, strict=True):
        if meta_doc.get("identity") != expected_graph_identity:
            sys.exit(
                f"{path} was exported for a different checkpoint or graph contract; "
                "re-export that graph family before creating this bundle")
    graphs = [g for meta_doc in meta_docs for g in meta_doc["graphs"]]
    # size_mb from the file ON DISK, not from _meta. The scratch files record what the
    # exporter wrote; a bundle can legitimately be rebuilt over graphs that were rewritten
    # since -- tools/fp16_weights.py halves every one of them -- and then _meta's sizes are
    # a bundle that misreports its own footprint by 2x. Which is exactly the number someone
    # consults to ask whether it fits on an 8 GB board. params stay from _meta: those are a
    # property of the graph, unchanged by a weight cast.
    for g in graphs:
        f = args.out_dir / str(g.get("file") or "")
        if f.is_file():
            # plus external data, if a converter wrote any: the .onnx alone is then a few kB
            ext = f.with_name(f.name + ".data")
            g["size_mb"] = round((f.stat().st_size
                                  + (ext.stat().st_size if ext.is_file() else 0)) / 1e6, 1)
    denoise_meta = next(
        (doc for path, doc in zip(metas, meta_docs, strict=True)
         if path.stem == "_meta_denoise"), {})
    denoise_input_mode = denoise_meta.get("denoise_input_mode", "x_t")
    missing = [n for n in GRAPHS if not (args.out_dir / f"_meta_{n}.json").exists()]
    if missing:
        print(f"  note: no graphs exported yet for {missing}")
    if tokens_per_view is None:
        vision_meta = args.out_dir / "_meta_vision.json"
        if vision_meta.exists():
            tokens_per_view = json.loads(vision_meta.read_text()).get("tokens_per_view")

    copy_processor_artifacts(args.checkpoint, args.out_dir, processor_contract)
    tokenizer_identity = materialize_tokenizer(tokenizer_source, args.out_dir)
    checkpoint_identity = {
        "source": str(args.checkpoint),
        "tree_sha256": checkpoint_tree_sha,
        "random_init": bool(args.random_init),
    }
    tasks = args.tasks or _dataset_tasks(args.checkpoint)
    fps = args.fps or _dataset_fps(args.checkpoint)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "fps": fps,
        "checkpoint": checkpoint_identity,
        "processor_contract": processor_contract,
        "tokenizer": tokenizer_identity,
        "domain_id": args.domain_id,
        "valid_views": args.valid_views,
        "num_image_views": cfg.get("num_image_views"),
        "lang_len": args.lang_len,
        "tokens_per_view": tokens_per_view,
        "chunk_size": cfg.get("chunk_size"),
        "num_denoising_steps": cfg.get("num_denoising_steps"),
        "denoise_input_mode": denoise_input_mode,
        "hidden_size": cfg.get("hidden_size"),
        "dim_time": cfg.get("dim_time"),
        "max_state_dim": cfg.get("max_state_dim"),
        "max_action_dim": processor_contract["action"]["model_dim"],
        "real_state_dim": processor_contract["state"]["dim"],
        "real_action_dim": processor_contract["action"]["dim"],
        "action_mode": cfg.get("action_mode"),
        # Both keys: policy.bundle_tasks prefers the list and falls back to the string, so
        # this serves a multi-task runtime and an older single-task one at once.
        "task": tasks[0] if tasks else None,
        "tasks": tasks,
        "policy_splits": args.policy_splits,
        "budget_gb": args.budget_gb,
        "graphs": graphs,
        "provenance": _provenance(args.checkpoint, args.opset),
    }
    (args.out_dir / "bundle.json").write_text(json.dumps(bundle, indent=2))
    total = sum(g["params"] for g in graphs)
    print(f"\n{len(graphs)} graphs, {total / 1e6:.1f}M params total")
    print(f"largest engine: {max(g['params'] for g in graphs) * 4 / 1e9:.2f} GB fp32")
    print(f"wrote {args.out_dir / 'bundle.json'}")
    if not tasks:
        print("!! no task strings — run_inference will refuse to start without --task.")
    else:
        print(f"   tasks: {tasks}")
    if not fps:
        print("!! no fps — run_inference will refuse to start without --fps.")
    else:
        print(f"   fps:   {fps}")
    if missing:
        print("skipping MANIFEST.sha256 — the bundle is incomplete "
              f"(no graphs for {missing})")
    else:
        n = write_manifest(args.out_dir)
        print(f"wrote {args.out_dir / 'MANIFEST.sha256'} ({n} files)\n"
              f"  verify after transfer:  cd <bundle> && sha256sum -c MANIFEST.sha256")


if __name__ == "__main__":
    main()
