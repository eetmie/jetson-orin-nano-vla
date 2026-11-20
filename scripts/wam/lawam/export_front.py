from pathlib import Path
import torch
from reference import fixture_obs, load_and_prepare
import sys
import numpy as np
from torch import nn

dry = "--dry" in sys.argv
# --fp32: FP32 graphs for vla_common.fp16_mixed; FP16 engines track an FP32 policy far
# closer than the stock BF16 autocast does.
fp32 = "--fp32" in sys.argv
DT = torch.float32 if fp32 else torch.bfloat16
directory = Path("bf-dry" if dry else "front32" if fp32 else "front")
directory.mkdir(exist_ok=True)


@torch.inference_mode()
def main():
    model, b = load_and_prepare(not dry)
    if fp32:
        from contextlib import nullcontext
        from lerobot.policies.lawam.vlas import lawam

        lawam._cuda_autocast = lambda dtype: nullcontext()
        torch.backends.cuda.matmul.allow_tf32 = False
        model.float()
    backend = model.model.policy_backend.cuda().eval()
    b = {k: v.cuda() for k, v in b.items()}
    torch.manual_seed(17)
    noise = torch.from_numpy(fixture_obs().noise).cuda()
    ll = backend.vlm.model.language_model
    record = {}

    def capture(mod, args, kwargs):
        record["cos"], record["sin"] = kwargs["position_embeddings"]
        record["attn"] = kwargs["attention_mask"]

    hook = ll.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    shared = backend._run_shared_encoding_infer(
        prepared_batch=b, source="bench", lam_features_with_no_grad=False
    )
    hook.remove()
    reference = model.predict_action_chunk(b, noise=noise)
    if not dry:
        torch.save(
            dict(
                batch={k: v.cpu() for k, v in b.items()},
                noise=noise.cpu(),
                reference=reference.cpu(),
                h_t=shared.h_t.cpu(),
                future=shared.h_t1_pred.cpu(),
                vlm=shared.h_vlm.cpu(),
            ),
            "stock.pt",
        )
    image = b["input_ids"][0] == backend.vlm.config.image_token_id
    img_indices = image.nonzero().flatten()
    fixed = backend.vlm.get_input_embeddings()(b["input_ids"]).clone()
    fixed[b["act_placeholder_mask"]] = backend.act_query.to(fixed)
    fixed[b["flow_placeholder_mask"]] = backend.flow_action_query.to(fixed)
    fixed = fixed.to(DT)
    ref_context = torch.cat(
        (
            shared.h_t.float(),
            shared.h_t1_pred.float(),
            backend.flow.enc_vlm(shared.h_vlm.float()),
        ),
        1,
    )

    backend.eval()
    torch.backends.mha.set_fastpath_enabled(False)

    def export(name, module, args, names, outs):
        inner = module.eval().cuda()

        class Boundary(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = inner

            def forward(self, *values):
                out = self.inner(*values)
                if isinstance(out, tuple):
                    return tuple(v.float() for v in out)
                return out.half() if name == "latent" else out.float()

        module = Boundary().eval()
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, cache_enabled=False, enabled=not fp32
        ):
            values = module(*args)
            if not dry or name in ("vis0", "lang0", "latent", "dino"):
                torch.onnx.export(
                    module,
                    args,
                    str(directory / (name + ".onnx")),
                    input_names=names,
                    output_names=outs,
                    opset_version=17,
                    dynamo=False,
                )
            print(name, flush=True)
            return values

    visual = backend.vlm.model.visual
    pixels = b["pixel_values"].float()
    vk = {}

    def capture_vision(mod, args, kwargs):
        vk.update(kwargs)
        vk["initial"] = args[0]

    h = visual.blocks[0].register_forward_pre_hook(capture_vision, with_kwargs=True)
    visual(torch.zeros_like(pixels).to(DT), b["image_grid_thw"])
    h.remove()
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights

    ix, wt = get_vision_interpolation_indices_and_weights(
        b["image_grid_thw"],
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.config.spatial_merge_size,
        kwargs={},
    )
    position_bias = (visual.pos_embed(ix) * wt[:, :, None]).sum(1).to(DT)

    class VisionGroup(nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i = i
            self.blocks = visual.blocks[6 * i : 6 * i + 6]
            self.merger = visual.deepstack_merger_list[i] if i < 3 else visual.merger
            if i == 0:
                self.patch = visual.patch_embed

        def forward(self, x):
            x = x.to(DT)
            if self.i == 0:
                x = self.patch(x) + position_bias
            for block in self.blocks:
                x = block(
                    x,
                    cu_seqlens=vk["cu_seqlens"],
                    position_embeddings=vk["position_embeddings"],
                    max_seqlen=vk.get("max_seqlen"),
                )
            return (x, self.merger(x)) if self.i < 3 else self.merger(x)

    vision = pixels
    deep = []
    for i in range(4):
        outs = [f"v{i}", f"d{i}"] if i < 3 else ["merged"]
        value = export(
            f"vis{i}",
            VisionGroup(i),
            (vision,),
            ["pixels" if i == 0 else f"v{i - 1}"],
            outs,
        )
        if i < 3:
            vision, d = value
            deep.append(d)
        else:
            vision = value

    class TextGroup(nn.Module):
        def __init__(self, start):
            super().__init__()
            self.start = start
            self.layers = ll.layers[start : start + 4]
            if start == 12:
                self.norm = ll.norm

        def forward(self, x, *deep):
            x = x.to(DT)
            if self.start == 0:
                x = torch.index_copy(fixed, 1, img_indices, x[None])
            for j, l in enumerate(self.layers):
                x = l(
                    x,
                    position_embeddings=(record["cos"], record["sin"]),
                    attention_mask=record["attn"]
                    if record["attn"] is not None
                    else None,
                )
                if self.start == 0 and j < 3:
                    x = x + torch.index_copy(
                        torch.zeros_like(x), 1, img_indices, deep[j][None].to(DT)
                    )
            if self.start == 12:
                x = self.norm(x)
            return x

    x = vision
    for i in range(4):
        args = (x, *deep) if i == 0 else (x,)
        names = ["merged", "d0", "d1", "d2"] if i == 0 else [f"l{i - 1}"]
        x = export(f"lang{i}", TextGroup(i * 4), args, names, [f"l{i}"])

    class Dino(nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.float32 if fp32 else torch.half
            self.dino = backend.lam.vision_encoder.to(self.dtype)

        def forward(self, image):
            with torch.autocast("cuda", enabled=False):
                return self.dino.encode(image.to(self.dtype), n=-2)[:, 0]

    primary = b["primary_image"].float()
    ht = export("dino", Dino(), (primary,), ["primary"], ["ht"])
    act_indices = b["act_placeholder_mask"][0].nonzero().flatten()

    class Latent(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = backend.vlm_to_lam
            self.decoder = backend.lam.decoder
            self.project = backend.flow.enc_vlm

        def forward(self, ht, l3):
            pred = self.q(torch.index_select(l3, 1, act_indices))
            future = self.decoder(ht, pred)[:, 0]
            with torch.autocast("cuda", enabled=False):
                language = self.project(l3.float())
            return torch.cat([ht.float(), future.float(), language], 1)

    ctx = export("latent", Latent(), (ht, x), ["ht", "l3"], ["context"])
    np.savez(
        directory / "input.npz",
        pixels=pixels.cpu().numpy(),
        primary=primary.cpu().numpy(),
        reference=reference.float().cpu().numpy(),
        noise=(noise * (torch.arange(50, device=noise.device) < 8)[None, :, None])
        .half()
        .cpu()
        .numpy(),
    )
    for label, actual, target in [
        ("vlm", x, shared.h_vlm),
        ("dino", ht, shared.h_t),
        ("context", ctx, ref_context),
    ]:
        aa = actual.flatten().double()
        bb = target.flatten().double()
        print(
            label,
            tuple(actual.shape),
            "finite",
            bool(torch.isfinite(actual).all()),
            "cos",
            float(torch.nn.functional.cosine_similarity(aa, bb, dim=0)),
            flush=True,
        )
    if not dry:
        torch.save(
            dict(context=ctx.cpu(), ref_context=ref_context.cpu()),
            directory / "front-check.pt",
        )


if __name__ == "__main__":
    main()
