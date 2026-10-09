# VENDORED from spark-projects @ 9e367cd — vla-onnx/groot/groot_split.py
"""GR00T N1.6 cut into traceable pieces sized for the 8 GB Orin.

One set of wrappers serves three jobs: the in-PyTorch split check (does the rearranged
model still produce the reference actions?), the ONNX export, and the parity reference.

Pipeline for one observation (V views, 4 denoising steps):

    host   tokens -> embed rows (mmap'd table, gather on CPU)
    vision_k   [V,3,H,W] -> [V,T,2048]       SigLIP2 + pixel unshuffle + mlp1, once
    host   scatter image tokens into the embedded sequence
    llm_k      [1,S,2048] -> [1,S,2048]      16 Qwen3 layers + norm, once
    cond       vlln(features), state_encoder(state)                    once
    dit_k      action encoder + 32 DiT blocks + decoder + Euler step,  4x

The sequence is right-padded to a fixed S. The LLM is causal, so trailing pads cannot
change any real token, and every DiT cross-attention masks them out: exact, not an
approximation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

MASK_NEG = -1.0e4


# --------------------------------------------------------------------------------------
# loading


def _per_image_eager(module, query, key, value, attention_mask, scaling, dropout=0.0,
                     seq_len_list=None, **kwargs):
    """SigLIP2 attention with flash-attn varlen semantics: each image attends to itself.

    The vendored eager path ignores `seq_len_list` and lets every packed image attend to
    every other one; only the flash kernel honours it. The checkpoint was trained with
    flash, so the reference has to be block-diagonal too.
    """
    outs = []
    start = 0
    lens = seq_len_list or [query.shape[2]]
    for n in lens:
        q = query[:, :, start:start + n]
        k = key[:, :, start:start + n]
        v = value[:, :, start:start + n]
        w = torch.softmax((q @ k.transpose(-1, -2)) * scaling, dim=-1, dtype=torch.float32)
        outs.append(w.to(q.dtype) @ v)
        start += n
    return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None


def load_policy(path: str, dtype=torch.float32):
    """Gr00tN1d6 with eager attention (no flash-attn), on CPU, in `dtype`."""

    # Eagle's __init__ asserts flash_attention_2, and transformers refuses that setting
    # when flash-attn is not installed. Let construction see "flash_attention_2" without
    # the import check, then flip every config to eager before anything runs.
    from transformers.modeling_utils import PreTrainedModel
    PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(
        lambda cls, config, *a, **kw: config)
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

    policy = Gr00tN1d6.from_pretrained(path, torch_dtype=dtype)
    policy = policy.to(dtype).eval()
    eagle = policy.backbone.model
    for m in eagle.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation_internal"):
            cfg._attn_implementation_internal = "eager"
    vit = eagle.vision_model
    sys.modules[type(vit.vision_model.encoder.layers[0].self_attn).__module__] \
        .eager_attention_forward = _per_image_eager
    return policy


# --------------------------------------------------------------------------------------
# contract


@dataclass
class Contract:
    embodiment: str
    embodiment_id: int
    views: int
    image_hw: tuple[int, int]
    tokens_per_image: int
    seq_len: int                 # padded S
    action_horizon: int = 50
    action_dim: int = 128
    state_dim: int = 128
    steps: int = 4
    buckets: int = 1000

    def timesteps(self) -> list[float]:
        return [float(int(i / self.steps * self.buckets)) for i in range(self.steps)]


# --------------------------------------------------------------------------------------
# vision


class VisionChunk(nn.Module):
    """SigLIP2 layers [a, b). First chunk patchifies and embeds; last adds post-LN,
    pixel unshuffle and the mlp1 projector."""

    def __init__(self, policy, a: int, b: int, image_hw: tuple[int, int]):
        super().__init__()
        eagle = policy.backbone.model
        tr = eagle.vision_model.vision_model
        self.first = a == 0
        self.last = b == len(tr.encoder.layers)
        self.layers = nn.ModuleList(tr.encoder.layers[a:b])
        self.patch = tr.embeddings.patch_size
        h, w = image_hw[0] // self.patch, image_hw[1] // self.patch
        self.grid = (h, w)
        if self.first:
            self.patch_embedding = tr.embeddings.patch_embedding
            emb = tr.embeddings
            pos = emb.position_embedding.weight.reshape(
                emb.position_embedding_size, emb.position_embedding_size, -1)
            pos = emb.resize_positional_embeddings(pos, torch.tensor([[h, w]]))[0]
            self.register_buffer("pos", pos.detach().clone(), persistent=False)
        if self.last:
            self.post_layernorm = tr.post_layernorm
            self.mlp1 = eagle.mlp1
            self.down = int(1 / eagle.downsample_ratio)

    def _attn(self, attn, x):
        b, n, c = x.shape
        q = attn.q_proj(x).view(b, n, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(b, n, attn.num_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(b, n, attn.num_heads, attn.head_dim).transpose(1, 2)
        w = torch.softmax((q @ k.transpose(-1, -2)) * attn.scale, dim=-1)
        return attn.out_proj((w @ v).transpose(1, 2).reshape(b, n, c))

    def forward(self, x):
        if self.first:
            # x: [V,3,H,W] normalized. Patchify exactly as convert_images_to_patches.
            v, ch, hh, ww = x.shape
            p = self.patch
            x = x.reshape(v, ch, hh // p, p, ww // p, p).permute(0, 2, 4, 3, 5, 1)
            x = x.reshape(v, (hh // p) * (ww // p), p * p * ch)
            x = self.patch_embedding(x) + self.pos
        for layer in self.layers:
            x = x + self._attn(layer.self_attn, layer.layer_norm1(x))
            x = x + layer.mlp(layer.layer_norm2(x))
        if self.last:
            x = self.post_layernorm(x)
            v, n, c = x.shape
            h, w = self.grid
            x = x.transpose(1, 2).reshape(v, c, h, w)
            x = F.pixel_unshuffle(x, self.down).flatten(2).transpose(1, 2)
            x = self.mlp1(x)
        return x


# --------------------------------------------------------------------------------------
# language model


class LlmChunk(nn.Module):
    """Qwen3 decoder layers [a, b) over a fixed, right-padded sequence. The causal mask
    and RoPE tables are constants because S is fixed. Last chunk applies the final norm
    (hidden_states[-1] of the HF model is post-norm)."""

    def __init__(self, policy, a: int, b: int, seq_len: int):
        super().__init__()
        lm = policy.backbone.model.language_model.model
        self.layers = nn.ModuleList(lm.layers[a:b])
        self.last = b == len(lm.layers)
        if self.last:
            self.norm = lm.norm
        pos = torch.arange(seq_len)[None]
        dummy = torch.zeros(1, seq_len, lm.config.hidden_size,
                            dtype=next(lm.parameters()).dtype)
        cos, sin = lm.rotary_emb(dummy, pos)
        self.register_buffer("cos", cos.detach().clone(), persistent=False)
        self.register_buffer("sin", sin.detach().clone(), persistent=False)
        mask = torch.full((seq_len, seq_len), MASK_NEG).triu(1)[None, None]
        self.register_buffer("mask", mask.to(dummy.dtype), persistent=False)
        self.register_buffer("pos", pos, persistent=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, attention_mask=self.mask, position_ids=self.pos,
                      position_embeddings=(self.cos, self.sin))[0]
        if self.last:
            x = self.norm(x)
        return x


# --------------------------------------------------------------------------------------
# action head


def _slice_cat_linear(lin, eid: int) -> nn.Linear:
    w = lin.W[eid].detach()               # [in, out]
    out = nn.Linear(w.shape[0], w.shape[1])
    out.weight.data = w.t().contiguous().clone()
    out.bias.data = lin.b[eid].detach().clone()
    return out


class Cond(nn.Module):
    """Loop-invariant conditioning: vlln on the backbone features, embodiment-sliced
    state encoder. Runs once per observation."""

    def __init__(self, policy, eid: int):
        super().__init__()
        head = policy.action_head
        self.vlln = head.vlln
        se = head.state_encoder
        self.s1 = _slice_cat_linear(se.layer1, eid)
        self.s2 = _slice_cat_linear(se.layer2, eid)

    def forward(self, features, state):
        return self.vlln(features), self.s2(F.relu(self.s1(state)))


class TimeEmb(nn.Module):
    """The two sinusoidal encodings of the denoising time t, kept as their own FP32 graph.

    sin(t * freq) with t up to 750 loses ~0.4 rad of argument when the product is FP16,
    and the mixed-FP16 pass would put it there; stock PyTorch computes both in FP32.
    Measured: inline FP16 sinusoids cost cos 0.992 / max 11.7 % of range on the chunk.
    """

    def __init__(self, policy):
        super().__init__()
        self.time_proj = policy.action_head.model.timestep_encoder.time_proj
        self.hidden = policy.action_head.action_encoder.hidden_size

    def forward(self, t):
        half = self.hidden // 2
        exponent = -torch.arange(half, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half)
        freqs = t.float().reshape(1, 1, 1) * exponent.exp()
        tau = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)    # [1,1,hidden]
        return self.time_proj(t), tau


class DitChunk(nn.Module):
    """DiT blocks [a, b) of one denoising step.

    First chunk: action encoder (sliced to the embodiment), position embedding, concat
    with the state token, timestep embedding. Last chunk: output AdaLN, proj_out,
    action decoder and the Euler update, so it returns the next action iterate.
    `temb` crosses every chunk boundary because every block's AdaLN needs it.
    """

    def __init__(self, policy, a: int, b: int, c: Contract):
        super().__init__()
        head = policy.action_head
        dit = head.model
        self.a, self.b = a, b
        self.first = a == 0
        self.last = b == len(dit.transformer_blocks)
        self.blocks = nn.ModuleList(dit.transformer_blocks[a:b])
        self.every = dit.attend_text_every_n_blocks
        self.dt = 1.0 / c.steps
        self.horizon = c.action_horizon
        eid = c.embodiment_id
        if self.first:
            enc = head.action_encoder
            self.w1 = _slice_cat_linear(enc.W1, eid)
            self.w2 = _slice_cat_linear(enc.W2, eid)
            self.w3 = _slice_cat_linear(enc.W3, eid)
            self.timestep_embedder = dit.timestep_encoder.timestep_embedder
            self.register_buffer(
                "pos_emb", head.position_embedding.weight[:c.action_horizon][None].detach().clone(),
                persistent=False)
        if self.last:
            self.norm_out = dit.norm_out
            self.proj_out_1 = dit.proj_out_1
            self.proj_out_2 = dit.proj_out_2
            dec = head.action_decoder
            self.d1 = _slice_cat_linear(dec.layer1, eid)
            self.d2 = _slice_cat_linear(dec.layer2, eid)

    def forward(self, *args):
        if self.first:
            actions, t_proj, tau, state_features, vl, text_bias, image_bias = args
            a_emb = self.w1(actions)
            x = torch.cat([a_emb, tau.to(a_emb.dtype).repeat(1, actions.shape[1], 1)], dim=-1)
            x = self.w2(x)
            x = self.w3(x * torch.sigmoid(x)) + self.pos_emb
            h = torch.cat([state_features, x], dim=1)
            temb = self.timestep_embedder(t_proj.to(a_emb.dtype))
        elif self.last:
            h, temb, vl, text_bias, image_bias, actions = args
        else:
            h, temb, vl, text_bias, image_bias = args
        for i, block in enumerate(self.blocks):
            idx = self.a + i
            if idx % 2 == 1:
                h = block(h, temb=temb)
            else:
                bias = text_bias if idx % (2 * self.every) == 0 else image_bias
                h = block(h, encoder_hidden_states=vl, encoder_attention_mask=bias, temb=temb)
        if not self.last:
            return (h, temb) if self.first else h
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        h = self.norm_out(h) * (1 + scale[:, None]) + shift[:, None]
        h = self.proj_out_2(h)
        v = self.d2(F.relu(self.d1(h)))[:, -self.horizon:]
        return actions + self.dt * v


# --------------------------------------------------------------------------------------
# planning


def plan(units: list[int], budget: int) -> list[tuple[int, int]]:
    """Greedy contiguous packing of per-layer param counts under `budget`."""
    out, start, acc = [], 0, 0
    for i, n in enumerate(units):
        if acc and acc + n > budget:
            out.append((start, i))
            start, acc = i, 0
        acc += n
    out.append((start, len(units)))
    return out


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def layer_plans(policy, budget: int) -> dict[str, list[tuple[int, int]]]:
    eagle = policy.backbone.model
    return {
        "vision": plan([n_params(l) for l in eagle.vision_model.vision_model.encoder.layers], budget),
        "llm": plan([n_params(l) for l in eagle.language_model.model.layers], budget),
        "dit": plan([n_params(b) for b in policy.action_head.model.transformer_blocks], budget),
    }


# --------------------------------------------------------------------------------------
# host-side glue (numpy-free here; the runtime mirrors it)


def biases(input_ids: torch.Tensor, attn: torch.Tensor, image_token: int):
    """Additive cross-attention biases [1,1,S] for text-attending and image-attending blocks."""
    img = input_ids == image_token
    valid = attn.bool()
    text_bias = torch.where((~img) & valid, 0.0, MASK_NEG)[:, None]
    image_bias = torch.where(img & valid, 0.0, MASK_NEG)[:, None]
    return text_bias, image_bias


def run_split(policy, plans, c: Contract, pixel, input_ids, attn, state, noise,
              dtype=torch.float32):
    """The whole split pipeline in PyTorch. Returns (actions, backbone_features)."""
    eagle = policy.backbone.model
    x = pixel.to(dtype)
    for a, b in plans["vision"]:
        x = VisionChunk(policy, a, b, c.image_hw)(x)
    emb = eagle.language_model.get_input_embeddings()(input_ids).to(dtype)
    sel = (input_ids == eagle.image_token_index)[0]
    emb[0, sel] = x.reshape(-1, x.shape[-1])
    h = emb
    for a, b in plans["llm"]:
        h = LlmChunk(policy, a, b, c.seq_len)(h)
    vl, sf = Cond(policy, c.embodiment_id)(h, state.to(dtype))
    tb, ib = biases(input_ids, attn, eagle.image_token_index)
    tb, ib = tb.to(dtype), ib.to(dtype)
    actions = noise.to(dtype)
    chunks = [DitChunk(policy, a, b, c) for a, b in plans["dit"]]
    time_emb = TimeEmb(policy).float()
    for t in c.timesteps():
        t_proj, tau = time_emb(torch.tensor([t]))
        hh, temb = chunks[0](actions, t_proj.to(dtype), tau.to(dtype), sf, vl, tb, ib)
        for ch in chunks[1:-1]:
            hh = ch(hh, temb, vl, tb, ib)
        actions = chunks[-1](hh, temb, vl, tb, ib, actions) if len(chunks) > 1 else hh
    return actions, h
