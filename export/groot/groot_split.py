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

import copy
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


def _is_cross(idx: int) -> bool:
    """Even DiT blocks cross-attend to the backbone features, odd ones self-attend."""
    return idx % 2 == 0


class _Half(nn.Module):
    """Stands in for a cross-attention block's to_k / to_v: the block is handed the
    precomputed [K | V] rows as its context and this picks one half, so the stock
    attention code (masking, SDPA, to_out) runs unchanged."""

    def __init__(self, i: int, inner: int):
        super().__init__()
        self.i, self.inner = i, inner

    def forward(self, x):
        return x[..., self.i * self.inner:(self.i + 1) * self.inner]


class ModChunk(nn.Module):
    """The AdaLN modulation of DiT blocks [a, b), plus the output AdaLN's on the last chunk.

    Each is linear(silu(temb)) with temb a function of the timestep alone, so for the
    fixed schedule they are constants: the runtime runs these graphs once at load and
    the per-step DiT graphs no longer carry those weights (~158 M of N1.6's 1.09 B).
    """

    def __init__(self, policy, a: int, b: int, last: bool):
        super().__init__()
        dit = policy.action_head.model
        self.timestep_embedder = dit.timestep_encoder.timestep_embedder
        self.lins = nn.ModuleList(dit.transformer_blocks[i].norm1.linear for i in range(a, b))
        self.proj_out_1 = dit.proj_out_1 if last else None
        self.out_names = [f"mod_{i}" for i in range(a, b)] + (["mod_out"] if last else [])

    def forward(self, t_proj):
        s = F.silu(self.timestep_embedder(t_proj))
        out = [lin(s) for lin in self.lins]
        if self.proj_out_1 is not None:
            out.append(self.proj_out_1(s))
        return tuple(out)


class KvChunk(nn.Module):
    """Cross-attention keys and values of the given DiT blocks, as [K | V] rows.

    They depend on the backbone features alone, so they run once per observation
    instead of once per denoising step.
    """

    def __init__(self, policy, idxs: list[int]):
        super().__init__()
        blocks = policy.action_head.model.transformer_blocks
        self.k = nn.ModuleList(blocks[i].attn1.to_k for i in idxs)
        self.v = nn.ModuleList(blocks[i].attn1.to_v for i in idxs)
        self.out_names = [f"kv_{i}" for i in idxs]

    def forward(self, vl):
        return tuple(torch.cat([k(vl), v(vl)], dim=-1) for k, v in zip(self.k, self.v))


class DitChunk(nn.Module):
    """DiT blocks [a, b) of one denoising step, fed precomputed modulation and K/V.

    First chunk: action encoder (sliced to the embodiment), position embedding, concat
    with the state token. Last chunk: output AdaLN, proj_out, action decoder and the
    Euler update, so it returns the next action iterate. Each block gets its AdaLN
    modulation (`mod_i`, from ModChunk) and, if it cross-attends, its [K | V] rows
    (`kv_i`, from KvChunk): the blocks are copies whose norm1.linear is an identity and
    whose to_k / to_v pick the halves, so the stock block code runs unchanged.
    Inputs and outputs are named in `in_names` / `out_names`.
    """

    def __init__(self, policy, a: int, b: int, c: Contract):
        super().__init__()
        head = policy.action_head
        dit = head.model
        self.a, self.b = a, b
        self.first = a == 0
        self.last = b == len(dit.transformer_blocks)
        self.every = dit.attend_text_every_n_blocks
        self.dt = 1.0 / c.steps
        self.horizon = c.action_horizon
        blocks = []
        for i in range(a, b):
            blk = copy.deepcopy(dit.transformer_blocks[i])
            blk.norm1.linear, blk.norm1.silu = nn.Identity(), nn.Identity()
            if _is_cross(i):
                inner = blk.attn1.to_q.out_features
                blk.attn1.to_k, blk.attn1.to_v = _Half(0, inner), _Half(1, inner)
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)
        eid = c.embodiment_id
        if self.first:
            enc = head.action_encoder
            self.w1 = _slice_cat_linear(enc.W1, eid)
            self.w2 = _slice_cat_linear(enc.W2, eid)
            self.w3 = _slice_cat_linear(enc.W3, eid)
            self.register_buffer(
                "pos_emb", head.position_embedding.weight[:c.action_horizon][None].detach().clone(),
                persistent=False)
        if self.last:
            self.norm_out = dit.norm_out
            self.proj_out_2 = dit.proj_out_2
            dec = head.action_decoder
            self.d1 = _slice_cat_linear(dec.layer1, eid)
            self.d2 = _slice_cat_linear(dec.layer2, eid)
        cross = [i for i in range(a, b) if _is_cross(i)]
        bias = []
        if any(i % (2 * self.every) == 0 for i in cross):
            bias.append("text_bias")
        if any(i % (2 * self.every) != 0 for i in cross):
            bias.append("image_bias")
        self.in_names = ((["actions", "tau", "state_features"] if self.first else ["h"])
                         + [f"mod_{i}" for i in range(a, b)]
                         + (["mod_out"] if self.last else [])
                         + [f"kv_{i}" for i in cross] + bias
                         + (["actions"] if self.last and not self.first else []))
        self.out_names = ["actions_next"] if self.last else ["h_out"]

    def example(self, c: Contract, hd: int, mod_dim: int, kv_dim: int, S: int) -> tuple:
        """Zero inputs in `in_names` order, for tracing."""
        shapes = {"actions": (1, c.action_horizon, c.action_dim), "tau": (1, 1, hd),
                  "state_features": (1, 1, hd), "h": (1, 1 + c.action_horizon, hd),
                  "mod_out": (1, 2 * hd), "text_bias": (1, 1, S), "image_bias": (1, 1, S)}
        def shape(n):
            if n.startswith("mod_"):
                return shapes.get(n, (1, mod_dim))
            if n.startswith("kv_"):
                return (1, S, kv_dim)
            return shapes[n]
        return tuple(torch.zeros(*shape(n)) for n in self.in_names)

    def forward(self, *args):
        x = dict(zip(self.in_names, args))
        if self.first:
            actions = x["actions"]
            a_emb = self.w1(actions)
            h = torch.cat([a_emb, x["tau"].to(a_emb.dtype).repeat(1, actions.shape[1], 1)], dim=-1)
            h = self.w2(h)
            h = self.w3(h * torch.sigmoid(h)) + self.pos_emb
            h = torch.cat([x["state_features"], h], dim=1)
        else:
            h = x["h"]
        for idx, block in zip(range(self.a, self.b), self.blocks):
            temb = x[f"mod_{idx}"]
            if not _is_cross(idx):
                h = block(h, temb=temb)
            else:
                bias = x["text_bias" if idx % (2 * self.every) == 0 else "image_bias"]
                h = block(h, encoder_hidden_states=x[f"kv_{idx}"],
                          encoder_attention_mask=bias, temb=temb)
        if not self.last:
            return h
        shift, scale = x["mod_out"].chunk(2, dim=1)
        h = self.norm_out(h) * (1 + scale[:, None]) + shift[:, None]
        h = self.proj_out_2(h)
        v = self.d2(F.relu(self.d1(h)))[:, -self.horizon:]
        return x["actions"] + self.dt * v


def add_action_graphs(add, policy, c: Contract, plans: dict, S: int, D: int) -> None:
    """The action head as graphs: time (FP32 sinusoids), mod_k (load-time constants),
    kv_k (once per observation) and dit_k (per step). Shared by both exporters."""
    dit = policy.action_head.model
    hd = policy.action_head.input_embedding_dim
    add("time", TimeEmb(policy).eval(), (torch.zeros(1),), ["t"], ["t_proj", "tau"])
    td = dit.timestep_encoder.timestep_embedder.linear_1.in_features
    for k, (lo, hi) in enumerate(plans["mod"]):
        m = ModChunk(policy, lo, hi, last=hi == len(dit.transformer_blocks)).eval()
        add(f"mod_{k}", m, (torch.zeros(1, td),), ["t_proj"], m.out_names)
    cross = [i for i in range(len(dit.transformer_blocks)) if _is_cross(i)]
    for k, (lo, hi) in enumerate(plans["kv"]):
        m = KvChunk(policy, cross[lo:hi]).eval()
        add(f"kv_{k}", m, (torch.zeros(1, S, D),), ["vl"], m.out_names)
    blk = dit.transformer_blocks[0]
    mod_dim, kv_dim = blk.norm1.linear.out_features, 2 * blk.attn1.to_k.out_features
    for k, (lo, hi) in enumerate(plans["dit"]):
        m = DitChunk(policy, lo, hi, c).eval()
        add(f"dit_{k}", m, m.example(c, hd, mod_dim, kv_dim, S), m.in_names, m.out_names)


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
        **action_plans(policy, budget),
    }


def action_plans(policy, budget: int) -> dict[str, list[tuple[int, int]]]:
    """mod / kv / dit groupings: the per-step DiT blocks without their AdaLN linears
    and cross-attention K/V projections, which live in the mod and kv graphs."""
    blocks = policy.action_head.model.transformer_blocks
    kv = [n_params(b.attn1.to_k) + n_params(b.attn1.to_v)
          for i, b in enumerate(blocks) if _is_cross(i)]
    step = [n_params(b) - n_params(b.norm1.linear)
            - (n_params(b.attn1.to_k) + n_params(b.attn1.to_v) if _is_cross(i) else 0)
            for i, b in enumerate(blocks)]
    return {
        "mod": plan([n_params(b.norm1.linear) for b in blocks], budget),
        "kv": plan(kv, budget),
        "dit": plan(step, budget),
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
    dit = policy.action_head.model
    pool = {"state_features": sf, "text_bias": tb, "image_bias": ib}
    cross = [i for i in range(len(dit.transformer_blocks)) if _is_cross(i)]
    for lo, hi in plans["kv"]:
        m = KvChunk(policy, cross[lo:hi])
        pool.update(zip(m.out_names, m(vl)))
    chunks = [DitChunk(policy, a, b, c) for a, b in plans["dit"]]
    mods = [ModChunk(policy, a, b, b == len(dit.transformer_blocks)) for a, b in plans["mod"]]
    time_emb = TimeEmb(policy).float()
    for t in c.timesteps():
        t_proj, tau = time_emb(torch.tensor([t]))
        pool["tau"] = tau.to(dtype)
        for m in mods:
            pool.update(zip(m.out_names, m(t_proj.to(dtype))))
        pool["actions"] = actions
        for ch in chunks:
            out = ch(*[pool[n] for n in ch.in_names])
            pool["h"] = out
        actions = pool["h"]
    return actions, h
