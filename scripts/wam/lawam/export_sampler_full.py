import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from lerobot.policies.lawam.vlas.flowmatching_expert import (
    ConditionalFlowMatchingConfig,
    ConditionalFlowMatchingHead,
)

root = Path("full")
root.mkdir(exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
cfg = json.loads(Path("ckpt/config.json").read_text())
options = ConditionalFlowMatchingConfig()
for k in vars(options):
    if "flow_" + k in cfg:
        setattr(options, k, cfg["flow_" + k])
head = ConditionalFlowMatchingHead(options).eval()
head.action_horizon, head.effective_action_horizon = 50, 8
prefix = "model.policy_backend.flow."
with safe_open("ckpt/model.safetensors", framework="pt", device="cpu") as checkpoint:
    state = {
        key[len(prefix) :]: checkpoint.get_tensor(key)
        for key in checkpoint.keys()
        if key.startswith(prefix)
    }
head.load_state_dict(state, strict=True, assign=True)
head.float().requires_grad_(False)
torch.manual_seed(71)


def rand(*shape):
    return torch.randn(*shape).half().float()


ht, future, vlm = rand(1, 256, 768), rand(1, 256, 768), rand(1, 204, 2048)
noise = rand(1, 50, 32)
idx = torch.tensor([25])
hz = torch.tensor([20.0])
# Only the first 8 of the 50 action tokens are valid. They never attend to the padded
# ones and every other op is per token, so the sampler runs on those 8 alone.
N = 8
with torch.inference_mode():
    ref = head.sample_actions_cfg(
        ht,
        future,
        vlm,
        torch.zeros(1, 32),
        torch.zeros(1, 32),
        hz,
        idx,
        1.0,
        10,
        (torch.arange(204) < 204)[None],
        noise=noise,
    )
    context = torch.cat((ht, future, head.enc_vlm(vlm)), dim=1)
    pos = head.time_encoder(torch.arange(N)[None] / 20.0)
    times = head.DiT.timestep_encoder(torch.arange(10) * 100)
    noise = noise[:, :N]
    np.savez(
        root / "fixture.npz",
        context=context.half().numpy(),
        pos=pos.half().numpy(),
        times=times.half().numpy(),
        noise=noise.half().numpy(),
        reference=ref.numpy(),
    )

blocks = head.DiT.transformer_blocks
# Cross-attention blocks (even index) alternate text (index % 4 == 0) and image tokens.
CROSS = [(i, slice(512, 716) if i % 4 == 0 else slice(0, 512)) for i in range(0, 16, 2)]


def split_heads(x, attn):
    return x.view(1, -1, attn.heads, x.shape[-1] // attn.heads).transpose(1, 2)


class KV(nn.Module):
    """Cross-attention keys and values: the context is fixed for all ten steps."""

    def forward(self, context):
        out = []
        for i, part in CROSS:
            a = blocks[i].attn1
            out += [split_heads(a.to_k(context[:, part]), a), split_heads(a.to_v(context[:, part]), a)]
        return tuple(out)


def cross_block(block, x, k, v, t, pos):
    # BasicTransformerBlock.forward with AttnProcessor2_0, keys and values precomputed.
    a = block.attn1
    h = block.norm1(x, t) + pos
    o = F.scaled_dot_product_attention(split_heads(a.to_q(h), a), k, v)
    x = x + a.to_out[0](o.transpose(1, 2).reshape(1, -1, a.inner_dim))
    return x + block.ff(block.norm3(x))


class Group(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i = i
        self.blocks = blocks[4 * i : 4 * i + 4]
        if i == 0:
            self.encoder = head.action_encoder
        if i == 3:
            self.norm = head.DiT.norm_out
            self.mod = head.DiT.proj_out_1
            self.proj = head.DiT.proj_out_2
            self.decoder = head.action_decoder

    def forward(self, x, kt, vt, ki, vi, t, pos, actions):
        if self.i == 0:
            x = self.encoder(actions, idx)
        kv = [(kt, vt), (ki, vi)]
        for j, block in enumerate(self.blocks):
            if j % 2:
                x = block(x, temb=t, hidden_positional_embeddings=pos)
            else:
                x = cross_block(block, x, *kv[j // 2], t, pos)
        if self.i == 3:
            shift, scale = self.mod(F.silu(t)).chunk(2, dim=-1)
            h = self.proj(self.norm(x) * (1 + scale[:, None]) + shift[:, None])
            return actions + 0.1 * self.decoder(h, idx)
        return x


GROUP_IN = ["x", "kt", "vt", "ki", "vi", "t", "pos", "actions"]
KV_OUT = [f"{n}{i}" for i in range(4) for n in ("kt", "vt", "ki", "vi")]
with torch.inference_mode():
    kv = KV()(context)
    actions = noise
    for step in range(10):
        x = torch.zeros(1, N, 1024)
        for i in range(4):
            x = Group(i)(x, *kv[4 * i : 4 * i + 4], times[step : step + 1], pos, actions)
        actions = x
    error = (actions - ref[:, :N]).abs().max().item()
    print("8-token sampler vs stock, max abs", error, flush=True)
    assert error < 1e-4
    graphs = [("kv", KV(), (context,), ["context"], KV_OUT)]
    graphs += [
        (f"g{i}", Group(i), (torch.zeros(1, N, 1024), *kv[4 * i : 4 * i + 4], times[:1], pos, noise), GROUP_IN, ["out"])
        for i in range(4)
    ]
    # FP32 graphs; vla_common.fp16_mixed --half-io '.*' makes them FP16 (README).
    for name, module, args, ins, outs in graphs:
        torch.onnx.export(
            module.eval(),
            args,
            str(root / f"{name}.onnx"),
            input_names=ins,
            output_names=outs,
            dynamo=False,
            opset_version=17,
        )
