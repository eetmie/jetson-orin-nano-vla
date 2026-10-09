# VENDORED from spark-projects @ c73dea0 — vla-onnx/groot/groot17_split.py
"""GR00T N1.7 cut into traceable pieces sized for the 8 GB Orin.

Same scheme as groot_split.py (N1.6), new backbone: Cosmos-Reason2-2B (Qwen3-VL).

    host   tokens -> embed rows (mmap'd table, gather on CPU)
    vision_k   [N,3,H,W] -> [N,T,2048]   Qwen3-VL ViT + merger; three chunks also emit
                                         a DeepStack feature [N,T,2048]
    host   scatter image tokens into the sequence, DeepStack features into [1,S,2048]
    llm_k      [1,S,2048] -> [1,S,2048]  16 decoder layers (MRoPE); layers 0-2 add the
                                         DeepStack features after their output
    cond_k     vlln + 4-layer VL self-attention + state encoder,             once
    dit_k      action encoder + 32 DiT blocks + decoder + Euler step,        4x

N images = views x 2 frames (now and ~1 s ago). Every image is encoded on its own (the
ViT attends within an image), so a runtime can cache a frame's vision output.

The sequence is right-padded to a fixed S: the LLM is causal, the VL self-attention
gets a key bias that hides the pads, and every DiT cross-attention masks them out.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from groot_split import MASK_NEG, plan, n_params  # noqa: F401  (re-exported)
from groot_split import (DitChunk, KvChunk, ModChunk, TimeEmb, _is_cross,  # noqa: E402
                         action_plans)

PATCH, MERGE = 16, 2


# --------------------------------------------------------------------------------------
# loading


def load_policy(path: str, vlm_files: str, dtype=torch.float32):
    """Gr00tN1d7 on CPU in `dtype`, eager attention, no download of the VLM weights.

    The model __init__ calls Qwen3VLForConditionalGeneration.from_pretrained(model_name),
    which would fetch Cosmos-Reason2-2B's own 2B weights only to overwrite them with the
    GR00T checkpoint. Build it from the config instead. `vlm_files` is where the config,
    tokenizer and processor files come from (the gated nvidia/Cosmos-Reason2-2B repo).
    """
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as proc_mod
    import gr00t.model.modules.qwen3_backbone as bb_mod

    def from_config(_name, **kw):
        cfg = AutoConfig.from_pretrained(vlm_files)
        cfg._attn_implementation = "eager"
        for sub in ("text_config", "vision_config"):
            getattr(cfg, sub)._attn_implementation = "eager"
        return Qwen3VLForConditionalGeneration._from_config(cfg, torch_dtype=dtype)

    bb_mod.Qwen3VLForConditionalGeneration = type(
        "Qwen3VLFromConfig", (), {"from_pretrained": staticmethod(from_config)})
    real_build = proc_mod.build_processor
    proc_mod.build_processor = lambda _name, kw: real_build(vlm_files, kw)

    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    policy = Gr00tN1d7.from_pretrained(path, torch_dtype=dtype)
    return policy.to(dtype).eval()


def vlm(policy):
    """The Qwen3VLModel inside the backbone (visual + language_model)."""
    return policy.backbone.model.model


# --------------------------------------------------------------------------------------
# contract


@dataclass
class Contract:
    embodiment: str
    embodiment_id: int
    views: int
    frames: int                  # images per view: history frame(s) + now
    image_hw: tuple[int, int]
    seq_len: int                 # padded S
    action_horizon: int = 40
    action_dim: int = 132
    state_dim: int = 132
    steps: int = 4
    buckets: int = 1000

    @property
    def images(self) -> int:
        return self.views * self.frames

    @property
    def grid(self) -> tuple[int, int]:
        return self.image_hw[0] // PATCH, self.image_hw[1] // PATCH

    @property
    def tokens_per_image(self) -> int:
        gh, gw = self.grid
        return gh * gw // (MERGE * MERGE)

    def timesteps(self) -> list[float]:
        return [float(int(i / self.steps * self.buckets)) for i in range(self.steps)]


# --------------------------------------------------------------------------------------
# vision


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class VisionChunk(nn.Module):
    """ViT blocks [a, b) over N independent images, batched.

    First chunk: patchify in the processor's merge-block order, patch embedding (the
    Conv3d over two identical temporal copies == one linear with the two kernel slices
    summed), interpolated position embedding. Every block whose index is a DeepStack
    layer also returns that layer's merger output. Last chunk adds the final merger.
    """

    def __init__(self, policy, a: int, b: int, c: Contract):
        super().__init__()
        vit = vlm(policy).visual
        self.first = a == 0
        self.last = b == len(vit.blocks)
        self.blocks = nn.ModuleList(vit.blocks[a:b])
        self.ds_at = {i - a: vit.deepstack_merger_list[k]
                      for k, i in enumerate(vit.deepstack_visual_indexes) if a <= i < b}
        self.ds_mergers = nn.ModuleList(self.ds_at.values())
        self.gh, self.gw = c.grid
        thw = torch.tensor([[1, self.gh, self.gw]])
        with torch.no_grad():
            if self.first:
                w = vit.patch_embed.proj.weight                       # [D,3,2,16,16]
                self.embed = nn.Linear(w.shape[1] * PATCH * PATCH, w.shape[0])
                self.embed.weight.data = w.sum(2).reshape(w.shape[0], -1).clone()
                self.embed.bias.data = vit.patch_embed.proj.bias.detach().clone()
                self.register_buffer("pos", vit.fast_pos_embed_interpolate(thw).detach().clone(),
                                     persistent=False)
            rot = vit.rot_pos_emb(thw)
            emb = torch.cat((rot, rot), dim=-1)
        self.register_buffer("cos", emb.cos()[:, None].float(), persistent=False)   # [P,1,hd]
        self.register_buffer("sin", emb.sin()[:, None].float(), persistent=False)
        if self.last:
            self.merger = vit.merger

    def _attn(self, attn, x):
        n, p, d = x.shape
        q, k, v = attn.qkv(x).reshape(n, p, 3, attn.num_heads, -1).unbind(2)   # [N,P,H,hd]
        q = q * self.cos + _rotate_half(q) * self.sin
        k = k * self.cos + _rotate_half(k) * self.sin
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        w = torch.softmax((q @ k.transpose(-1, -2)) * attn.scaling, dim=-1)
        return attn.proj((w @ v).transpose(1, 2).reshape(n, p, d))

    def forward(self, x):
        if self.first:
            # x: [N,3,H,W] normalized -> [N, P, 3*16*16], P in merge-block order exactly
            # as Qwen2VLImageProcessor flattens it.
            n = x.shape[0]
            gh, gw = self.gh, self.gw
            x = x.reshape(n, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
            x = x.permute(0, 2, 5, 3, 6, 1, 4, 7).reshape(n, gh * gw, 3 * PATCH * PATCH)
            x = self.embed(x) + self.pos
        ds = []
        for i, blk in enumerate(self.blocks):
            x = x + self._attn(blk.attn, blk.norm1(x))
            x = x + blk.mlp(blk.norm2(x))
            if i in self.ds_at:
                m = self.ds_at[i]                                     # postshuffle norm
                n, p, d = x.shape
                ds.append(m.linear_fc2(m.act_fn(m.linear_fc1(
                    m.norm(x.reshape(n, p // 4, 4 * d))))))
        out = x
        if self.last:
            m = self.merger
            n, p, d = x.shape
            out = m.linear_fc2(m.act_fn(m.linear_fc1(m.norm(x).reshape(n, p // 4, 4 * d))))
        return (out, *ds)


# --------------------------------------------------------------------------------------
# language model


def mrope(policy, input_ids: torch.Tensor, n_real: int, c: Contract):
    """cos/sin [1,S,128] of Qwen3-VL's interleaved MRoPE for this fixed token layout.

    Positions of the real tokens are exactly get_rope_index's; the pads continue the
    text positions (they are never attended to by a real token).
    """
    m = vlm(policy)
    thw = torch.tensor([[1, *c.grid]] * c.images)
    pos, _ = m.get_rope_index(input_ids[:, :n_real], thw, None)
    tail = pos[:, :, -1:].max() + 1 + torch.arange(c.seq_len - n_real)
    pos = torch.cat([pos, tail.view(1, 1, -1).expand(3, 1, -1)], dim=2)
    lm = m.language_model
    dummy = torch.zeros(1, c.seq_len, lm.config.hidden_size)
    return lm.rotary_emb(dummy, pos)


class LlmChunk(nn.Module):
    """Decoder layers [a, b) over the right-padded sequence. Layer i < 3 adds DeepStack
    feature i (zero outside image positions) to its output, as Qwen3VLTextModel does.

    No final norm: stock backbone_features is Qwen3VLForConditionalGeneration's
    hidden_states[-1], which is the last decoder layer's output BEFORE the model's final
    RMSNorm (measured; transformers 4.57.3 does not tie it to last_hidden_state at that
    level). It still carries the ~1.5e4 massive activation into vlln.
    """

    def __init__(self, policy, a: int, b: int, c: Contract, cos, sin, n_deepstack: int):
        super().__init__()
        lm = vlm(policy).language_model
        self.layers = nn.ModuleList(lm.layers[a:b])
        self.ds_layers = [i - a for i in range(a, b) if i < n_deepstack]
        self.last = b == len(lm.layers)
        self.register_buffer("cos", cos.detach().clone(), persistent=False)
        self.register_buffer("sin", sin.detach().clone(), persistent=False)
        mask = torch.full((c.seq_len, c.seq_len), MASK_NEG).triu(1)[None, None]
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, *ds):
        it = iter(ds)
        for i, layer in enumerate(self.layers):
            x = layer(x, position_embeddings=(self.cos, self.sin), attention_mask=self.mask)
            if isinstance(x, tuple):
                x = x[0]
            if i in self.ds_layers:
                x = x + next(it)
        return x


# --------------------------------------------------------------------------------------
# action head


class Cond(nn.Module):
    """Loop-invariant conditioning, run once per observation: vlln, the VL
    self-attention blocks [a, b) with the pads hidden from every key, and (first chunk)
    the embodiment-sliced state encoder."""

    def __init__(self, policy, a: int, b: int, eid: int):
        super().__init__()
        from groot_split import _slice_cat_linear

        head = policy.action_head
        self.first = a == 0
        self.blocks = nn.ModuleList(head.vl_self_attention.transformer_blocks[a:b])
        if self.first:
            self.vlln = head.vlln
            se = head.state_encoder
            self.s1 = _slice_cat_linear(se.layer1, eid)
            self.s2 = _slice_cat_linear(se.layer2, eid)

    def forward(self, x, pad_bias, state=None):
        if self.first:
            x = self.vlln(x)
        for blk in self.blocks:
            x = blk(x, attention_mask=pad_bias)
        if self.first:
            return x, self.s2(F.relu(self.s1(state)))
        return x


def layer_plans(policy, budget: int) -> dict[str, list[tuple[int, int]]]:
    m = vlm(policy)
    head = policy.action_head
    vit = m.visual
    # A DeepStack layer's merger (~25 M) and the final merger ride in that layer's chunk.
    extra = {i: n_params(vit.deepstack_merger_list[k])
             for k, i in enumerate(vit.deepstack_visual_indexes)}
    extra[len(vit.blocks) - 1] = extra.get(len(vit.blocks) - 1, 0) + n_params(vit.merger)
    return {
        "vision": plan([n_params(b) + extra.get(i, 0) for i, b in enumerate(vit.blocks)], budget),
        "llm": plan([n_params(l) for l in m.language_model.layers], budget),
        "cond": plan([n_params(b) for b in head.vl_self_attention.transformer_blocks], budget),
        **action_plans(policy, budget),
    }


# --------------------------------------------------------------------------------------
# host-side glue


def masks(input_ids: torch.Tensor, n_real: int, image_token: int):
    """Additive biases [1,1,S]: text-attending / image-attending DiT blocks, and the pad
    key bias of the VL self-attention."""
    s = input_ids.shape[1]
    img = input_ids == image_token
    valid = (torch.arange(s) < n_real)[None]
    text_bias = torch.where((~img) & valid, 0.0, MASK_NEG)[:, None]
    image_bias = torch.where(img & valid, 0.0, MASK_NEG)[:, None]
    pad_bias = torch.where(valid, 0.0, MASK_NEG)[:, None]
    return text_bias, image_bias, pad_bias


def run_split(policy, plans, c: Contract, pixel, input_ids, n_real, state, noise):
    """The whole split pipeline in PyTorch FP32. pixel [N,3,H,W] in prompt image order.
    Returns (actions, vl): vl is the conditioning after vlln + VL self-attention, which
    stock get_action writes back into backbone_output["backbone_features"]."""
    from groot_split import DitChunk, TimeEmb

    m = vlm(policy)
    tok = m.config.image_token_id
    x, ds = pixel, []
    for a, b in plans["vision"]:
        x, *d = VisionChunk(policy, a, b, c)(x)
        ds += d
    h = m.language_model.embed_tokens(input_ids).clone()
    sel = (input_ids == tok)[0]
    h[0, sel] = x.reshape(-1, x.shape[-1])
    full = []
    for d in ds:
        z = torch.zeros_like(h)
        z[0, sel] = d.reshape(-1, d.shape[-1])
        full.append(z)
    cos, sin = mrope(policy, input_ids, n_real, c)
    for a, b in plans["llm"]:
        ch = LlmChunk(policy, a, b, c, cos, sin, len(full))
        h = ch(h, *[full[a + i] for i in ch.ds_layers])
    tb, ib, pb = masks(input_ids, n_real, tok)
    vl = h
    for a, b in plans["cond"]:
        ch = Cond(policy, a, b, c.embodiment_id)
        if ch.first:
            vl, sf = ch(vl, pb, state)
        else:
            vl = ch(vl, pb)
    actions = noise
    dit = policy.action_head.model
    pool = {"state_features": sf, "text_bias": tb, "image_bias": ib}
    cross = [i for i in range(len(dit.transformer_blocks)) if _is_cross(i)]
    for lo, hi in plans["kv"]:
        m = KvChunk(policy, cross[lo:hi])
        pool.update(zip(m.out_names, m(vl)))
    chunks = [DitChunk(policy, a, b, c) for a, b in plans["dit"]]
    mods = [ModChunk(policy, a, b, b == len(dit.transformer_blocks)) for a, b in plans["mod"]]
    time_emb = TimeEmb(policy)
    for t in c.timesteps():
        t_proj, tau = time_emb(torch.tensor([t]))
        pool["tau"] = tau
        for m in mods:
            pool.update(zip(m.out_names, m(t_proj)))
        pool["actions"] = actions
        for ch in chunks:
            pool["h"] = ch(*[pool[n] for n in ch.in_names])
        actions = pool["h"]
    return actions, vl
