import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors import safe_open
from lerobot.policies.lawam.vlas.flowmatching_expert import (
    ConditionalFlowMatchingConfig,
    ConditionalFlowMatchingHead,
)

root = Path("full")
root.mkdir(exist_ok=True)
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
torch.manual_seed(71)


def rand(*shape):
    return torch.randn(*shape).half().float()


ht, future, vlm = rand(1, 256, 768), rand(1, 256, 768), rand(1, 204, 2048)
noise = rand(1, 50, 32)
idx = torch.tensor([25])
hz = torch.tensor([20.0])
valid = (torch.arange(50) < 8)[None]
language = (torch.arange(204) < 204)[None]
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
        language,
        noise=noise,
    )
    context = torch.cat((ht, future, head.enc_vlm(vlm)), dim=1).half()
    pos = (head.time_encoder(torch.arange(50)[None] / 20.0) * valid[:, :, None]).half()
    times = head.DiT.timestep_encoder(torch.arange(10) * 100).half()
    noise = (noise * valid[:, :, None]).half()
    np.savez(
        root / "fixture.npz",
        context=context.numpy(),
        pos=pos.numpy(),
        times=times.numpy(),
        noise=noise.numpy(),
        reference=ref.numpy(),
    )


class Group(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i = i
        self.blocks = head.DiT.transformer_blocks[4 * i : 4 * i + 4]
        if i == 0:
            self.encoder = head.action_encoder
        if i == 3:
            self.norm = head.DiT.norm_out
            self.mod = head.DiT.proj_out_1
            self.proj = head.DiT.proj_out_2
            self.decoder = head.action_decoder

    def forward(self, x, context, t, pos, actions):
        if self.i == 0:
            x = self.encoder(actions, idx) * valid[:, :, None]
        for j, block in enumerate(self.blocks):
            if j % 2 == 1:
                x = block(
                    x, attention_mask=valid, temb=t, hidden_positional_embeddings=pos
                )
            else:
                # Select the valid context for this mask before cross-attention.
                c = context[:, 512:716] if j == 0 else context[:, :512]
                x = block(
                    x, encoder_hidden_states=c, temb=t, hidden_positional_embeddings=pos
                )
        if self.i == 3:
            shift, scale = self.mod(torch.nn.functional.silu(t)).chunk(2, dim=-1)
            h = self.proj(self.norm(x) * (1 + scale[:, None]) + shift[:, None])
            return (actions + 0.1 * self.decoder(h, idx)) * valid[:, :, None]
        return x


head.half()
with torch.inference_mode():
    x = torch.zeros(1, 50, 1024, dtype=torch.float16)
    for i in range(4):
        group = Group(i).eval()
        args = (x, context, times[:1], pos, noise)
        torch.onnx.export(
            group,
            args,
            str(root / f"g{i}.onnx"),
            input_names=["x", "context", "t", "pos", "actions"],
            output_names=["out"],
            dynamo=False,
            opset_version=17,
        )
        x = group(*args)
