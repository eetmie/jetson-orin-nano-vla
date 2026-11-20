"""Stock-policy action chunks for ring observations, with the noise run_full.py draws.

Writes pack/ring-reference.npz: "<index>" is the FP32 policy (the config default) and
"bf16/<index>" the same policy stored in BF16, both normalized actions [16, 10].
"""

from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from lerobot.policies.cosmos3.modeling_cosmos3 import Cosmos3EdgePolicy, FlowState
from lerobot.policies.cosmos3.configuration_cosmos3 import Cosmos3EdgeConfig
from lerobot.policies.cosmos3.processor_cosmos3 import (
    Cosmos3VideoCanvasProcessorStep as Canvas,
    Cosmos3PromptProcessorStep as Prompt,
    Cosmos3TokenizerProcessorStep as Tokenizer,
)
from bench.obs import SyntheticObs

INDICES = (0, 1, 5, 13)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
src = SyntheticObs(
    task="Pick up the cube.", chunk_size=16, state_dim=8, max_action_dim=64, h=256, w=256, n_views=2
)
state = load_file("ckpt/model.safetensors")
out = {}
for dtype in (torch.float32, torch.bfloat16):
    cfg = Cosmos3EdgeConfig.from_pretrained("ckpt")
    cfg.gradient_checkpointing = False
    cfg.dtype = dtype
    policy = Cosmos3EdgePolicy(cfg).eval()
    policy.load_state_dict(state, strict=True)
    policy.cuda()
    cam = Canvas(["a", "b"], (256, 256), 20, 16)
    prompt = Prompt(cfg.view_description, 20, 16, False)
    tok = Tokenizer(str(Path("ckpt/tokenizer").resolve()))
    for index in INDICES:
        obs = src[index]
        frames = torch.from_numpy(np.concatenate(obs.images, axis=1)).permute(2, 0, 1)[None, None]
        canvas, hw = cam.fit_canvas(frames)
        H, W = canvas.shape[-2:]
        pixels = (canvas.permute(0, 2, 1, 3, 4).float() / 127.5 - 1).cuda()
        ids, mask = tok.encode([prompt.caption(obs.task, H, W, 0, 20)])
        grid = torch.tensor([[5, hw[0] // 16, hw[1] // 16]], device="cuda")
        T, gh, gw = 5, hw[0] // 16, hw[1] // 16
        # run_full.py: patchified video noise from this generator, actions from the ring.
        rng = np.random.default_rng([1234, index, 11])
        video = rng.standard_normal((1, T, (gh // 2) * (gw // 2), 192), dtype=np.float32)
        net = policy.model.network
        video = net.unpatchify(torch.from_numpy(video).reshape(1, -1, 192).cuda(), (1, 48, T, gh, gw))
        actions = torch.from_numpy(obs.noise).float().cuda()
        actions[:, :, 10:] = 0
        with torch.inference_mode():
            chunk, _ = policy.model.generate(
                input_ids=ids.cuda(),
                pixel_values_videos_gen=pixels,
                video_grid_thw_gen=grid,
                second_per_grid_ts_gen=torch.tensor([0.2], device="cuda"),
                attention_mask=mask.cuda(),
                noise=FlowState(video, actions),
            )
        key = str(index) if dtype == torch.float32 else f"bf16/{index}"
        out[key] = chunk[0, :, :10].float().cpu().numpy()
        print(key, out[key][0, :4], flush=True)
    del policy
    torch.cuda.empty_cache()
for index in INDICES:
    a, b = out[str(index)].ravel().astype("f8"), out[f"bf16/{index}"].ravel().astype("f8")
    print(index, "bf16 vs fp32 cos", a @ b / np.linalg.norm(a) / np.linalg.norm(b))
Path("pack").mkdir(exist_ok=True)
np.savez("pack/ring-reference.npz", **out)
