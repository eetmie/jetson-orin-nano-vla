import json
from pathlib import Path
import torch
from bench.obs import SyntheticObs
from lerobot.policies.lawam.processor_lawam import (
    LaWAMQwenInputsProcessorStep,
    LaWAMPrepareBatchProcessorStep,
    TransitionKey,
)
from reference import load_and_prepare
import numpy as np
import argparse
from safetensors.numpy import load_file

parser = argparse.ArgumentParser()
# fp32: the whole policy in FP32 with SDPA, the reference the FP16 engines are held to.
parser.add_argument("--attention", choices=["eager", "sdpa", "fp32"], default="eager")
attention = parser.parse_args().attention
stats = load_file("ckpt/policy_postprocessor_step_2_unnormalizer_processor.safetensors")
np.savez("full/post.npz", lo=stats["action.min"], hi=stats["action.max"])
with torch.inference_mode():
    policy, _ = load_and_prepare(attention="sdpa" if attention == "fp32" else attention)
    policy.cuda().eval()
    if attention == "fp32":
        from contextlib import nullcontext
        from lerobot.policies.lawam.vlas import lawam

        lawam._cuda_autocast = lambda dtype: nullcontext()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        policy.float()
    d = json.loads(Path("ckpt/policy_preprocessor.json").read_text())["steps"]
    opt = next(s["config"] for s in d if s["registry_name"] == "lawam_qwen_inputs")
    opt["model_id"] = policy.config.base_vlm
    proc = LaWAMQwenInputsProcessorStep(**opt)
    finish = LaWAMPrepareBatchProcessorStep(**d[-1]["config"])
    source = SyntheticObs(
        task="Pick up the cube.", chunk_size=50, state_dim=32, n_views=2, h=256, w=256
    )
    refs = {}
    for index in (0, 1, 5, 13):
        obs = source[index]
        images = {
            k: torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
            for k, img in zip(
                ("observation.images.image", "observation.images.image2"), obs.images
            )
        }
        tr = proc(
            {
                TransitionKey.OBSERVATION: images,
                TransitionKey.COMPLEMENTARY_DATA: {"task": [obs.task]},
            }
        )
        batch = finish(tr)[TransitionKey.COMPLEMENTARY_DATA]
        batch = {k: v.cuda() for k, v in batch.items()}
        refs[str(index)] = (
            policy.predict_action_chunk(batch, noise=torch.from_numpy(obs.noise).cuda())
            .float()
            .cpu()
            .numpy()
        )
    filename = {
        "eager": "ring-reference.npz",
        "sdpa": "ring-sdpa-reference.npz",
        "fp32": "ring-fp32-reference.npz",
    }[attention]
    np.savez(Path("full") / filename, **refs)
