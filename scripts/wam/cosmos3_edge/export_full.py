"""Export fixed-prompt Cosmos3 Edge LIBERO inference for an initial single-board test."""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from lerobot.policies.cosmos3.modeling_cosmos3 import Cosmos3EdgePolicy, rotate_half
from lerobot.policies.cosmos3.configuration_cosmos3 import Cosmos3EdgeConfig
from lerobot.policies.cosmos3.processor_cosmos3 import (
    Cosmos3VideoCanvasProcessorStep as Canvas,
    Cosmos3PromptProcessorStep as Prompt,
    Cosmos3TokenizerProcessorStep as Tokenizer,
)


class GenBlock(nn.Module):
    def __init__(self, layer, kg, vg, cos, sin):
        super().__init__()
        self.norm1 = layer.input_layernorm_gen
        self.norm2 = layer.post_attention_layernorm_gen
        self.mlp = layer.mlp_gen
        a = layer.self_attn
        self.q = a.q_proj_gen
        self.k = a.k_proj_gen
        self.v = a.v_proj_gen
        self.out = a.o_proj_gen
        self.qn = a.q_norm_gen
        self.kn = a.k_norm_gen
        for name, tensor in [
            ("uk", kg),
            ("uv", vg),
            ("cos", cos[:, None]),
            ("sin", sin[:, None]),
        ]:
            self.register_buffer(name, tensor.contiguous())

    def forward(self, x):
        h = self.norm1(x)
        q = self.qn(self.q(h).reshape(1, -1, 16, 128)).transpose(1, 2)
        k = self.kn(self.k(h).reshape(1, -1, 8, 128)).transpose(1, 2)
        q = q * self.cos + rotate_half(q) * self.sin
        k = k * self.cos + rotate_half(k) * self.sin
        v = self.v(h).reshape(1, -1, 8, 128).transpose(1, 2)
        k = torch.cat((self.uk, k), dim=2).repeat_interleave(2, dim=1)
        v = torch.cat((self.uv, v), dim=2).repeat_interleave(2, dim=1)
        # Keep softmax reductions in float32 on both architectures.
        s = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(h.dtype)
        x = x + self.out(s.transpose(1, 2).reshape(1, -1, 2048))
        return x + self.mlp(self.norm2(x))


class Group(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x.to(torch.bfloat16)).float()


class First(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.video = n.vae2llm
        self.register_buffer(
            "aw", n.action2llm.fc.weight[5].reshape(64, 2048).contiguous()
        )
        self.register_buffer("ab", n.action2llm.bias.weight[5])
        self.mod = n.action_modality_embed

    def forward(self, patches, actions, temb):
        v = self.video(patches.to(self.video.weight.dtype))
        a = self.mod(actions.to(self.aw.dtype) @ self.aw + self.ab)
        v = torch.cat((v[:, :1], v[:, 1:] + temb.to(v.dtype)[:, None, None, :]), 1)
        return torch.cat((v.flatten(1, 2), a + temb.to(a.dtype)[:, None, :]), 1).float()


class Last(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.norm = n.mot_backbone.model.language_model.norm_gen
        self.video = n.llm2vae
        self.register_buffer(
            "aw", n.llm2action.fc.weight[5].reshape(2048, 64).contiguous()
        )
        self.register_buffer("ab", n.llm2action.bias.weight[5])

    def forward(self, x):
        h = self.norm(x.bfloat16())
        return self.video(h[:, :-16].to(self.video.weight.dtype)).float(), (
            h[:, -16:].to(self.aw.dtype) @ self.aw + self.ab
        ).float()


class Encoder(nn.Module):
    def __init__(self, n, grid):
        super().__init__()
        self.vae = n.vae
        self.hw = tuple(grid[0, 1:].tolist())

    def forward(self, canvas):
        z = self.vae.encode(canvas.to(self.vae.dtype)).latent_dist.mode()
        m = z.new_tensor(self.vae.config.latents_mean)[None, :, None, None, None]
        s = z.new_tensor(self.vae.config.latents_std)[
            None, :, None, None, None
        ].reciprocal()
        return ((z - m) * s)[:, :, :, : self.hw[0], : self.hw[1]].float()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only-vae", action="store_true")
    args = ap.parse_args()
    out = Path("dry" if args.dry else "pack")
    out.mkdir(exist_ok=True)
    cfg = Cosmos3EdgeConfig.from_pretrained("ckpt")
    cfg.gradient_checkpointing = False
    cfg.dtype = torch.bfloat16
    policy = Cosmos3EdgePolicy(cfg).eval()
    if not args.dry:
        policy.load_state_dict(load_file("ckpt/model.safetensors"), strict=True)
    n = policy.model.network.cuda().eval()
    ll = n.mot_backbone.model.language_model
    cam = Canvas(["a", "b"], (256, 256), 20, 16)
    prompt = Prompt(cfg.view_description, 20, 16, False)
    tok = Tokenizer(str(Path("ckpt/tokenizer").resolve()))
    rng = np.random.default_rng(123)
    cameras = rng.integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
    frames = torch.from_numpy(np.concatenate(cameras, axis=1)).permute(2, 0, 1)[
        None, None
    ]
    canvas, hw = cam.fit_canvas(frames)
    H, W = canvas.shape[-2:]
    pixels = (canvas.permute(0, 2, 1, 3, 4).float() / 127.5 - 1).cuda()
    ids, mask = tok.encode([prompt.caption("Pick up the cube.", H, W, 0, 20)])
    ids, mask = ids.cuda(), mask.cuda()
    grid = torch.tensor([[5, hw[0] // 16, hw[1] // 16]], device="cuda")
    nt = 5 * ((hw[0] // 16 + 1) // 2) * ((hw[1] // 16 + 1) // 2)
    print("profile", H, W, hw, nt, ids.shape, flush=True)

    def export(name, module, inputs, ins, outs):
        dest = out / (name + ".onnx")
        if args.dry and name not in ("vae", "group00", "first", "last"):
            return
        if dest.exists():
            return
        temp = dest.with_suffix(".partial")
        torch.onnx.export(
            module.eval(),
            inputs,
            str(temp),
            opset_version=17,
            input_names=ins,
            output_names=outs,
            dynamo=False,
        )
        temp.rename(dest)
        print("exported", name, dest.stat().st_size, flush=True)

    # The first observed frame alone is encoded; later latent frames start as noise.
    vae = Encoder(n, grid)
    export("vae", vae, (pixels,), ["canvas"], ["condition"])
    if args.only_vae:
        return
    pos = n.get_rope_index(
        ids, grid, torch.tensor([0.2], device="cuda"), 16, attention_mask=mask
    )
    u = ll.embed_tokens(ids)
    g = torch.zeros(1, nt + 16, 2048, dtype=u.dtype, device="cuda")
    cos, sin = ll.rotary_emb(g, pos)
    cu, cg = cos.split((ids.shape[1], nt + 16), 1)
    su, sg = sin.split((ids.shape[1], nt + 16), 1)
    assert mask.all()
    causal = torch.ones(
        1, 1, u.shape[1], u.shape[1], device="cuda", dtype=torch.bool
    ).tril()
    blocks = []
    for i, layer in enumerate(ll.layers):
        a = layer.self_attn
        h = layer.input_layernorm(u)
        ku = a.k_norm_und_for_gen(a.k_proj(h).reshape(1, -1, 8, 128).transpose(1, 2))
        ku = ku * cu[:, None] + rotate_half(ku) * su[:, None]
        vu = a.v_proj(h).reshape(1, -1, 8, 128).transpose(1, 2)
        blocks.append(GenBlock(layer, ku, vu, cg, sg))
        u, g = layer(u, g, (cu, su), (cg, sg), causal, None)
        if len(blocks) == 2:
            export(
                f"group{i // 2:02d}", Group(blocks), (g.float(),), ["hidden"], ["out"]
            )
            blocks = []
    nv = nt // 5
    pp = torch.zeros(1, 5, nv, 192, device="cuda")
    aa = torch.zeros(1, 16, 64, device="cuda")
    scheduler = policy.model.make_scheduler(torch.device("cpu"))
    ts = n.embed_timesteps(scheduler.timesteps.cuda()).float()
    export("first", First(n), (pp, aa, ts[:1]), ["patches", "actions", "time"], ["out"])
    export(
        "last", Last(n), (g.float(),), ["hidden"], ["video_velocity", "action_velocity"]
    )
    # UniPC is linear in the starting state and velocity history for this scheduler.
    # Propagating basis vectors gives the exact, step-specific trajectory coefficients.
    basis = torch.eye(31)
    x = basis[:1]
    rows = []
    for i, t in enumerate(scheduler.timesteps):
        x = scheduler.step(basis[i + 1 : i + 2], t, x).prev_sample
        rows.append(x.numpy().copy())
    coeff = np.concatenate(rows)
    assert np.isfinite(coeff).all()
    np.savez(
        out / "fixture.npz",
        canvas=pixels.cpu().numpy(),
        times=ts.cpu().numpy(),
        coeff=coeff,
        grid=grid.cpu().numpy(),
        camera_rgb=cameras,
    )
    print("complete", flush=True)


if __name__ == "__main__":
    main()
