from pathlib import Path
from huggingface_hub import snapshot_download
from bench.obs import SyntheticObs


def fixture_obs():
    return SyntheticObs(
        task="Pick up the cube.",
        chunk_size=50,
        state_dim=32,
        max_action_dim=32,
        n_views=2,
        h=256,
        w=256,
    )[0]


import json
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from lerobot.policies.lawam import modeling_lawam as lm
from lerobot.policies.lawam.vlas import lawam
from lerobot.policies.lawam.processor_lawam import (
    LaWAMQwenInputsProcessorStep,
    LaWAMPrepareBatchProcessorStep,
    TransitionKey,
)


def load_and_prepare(weights=True, attention="eager"):
    p = snapshot_download(
        "Qwen/Qwen3-VL-2B-Instruct",
        revision="89644892e4d85e24eaac8bacfd4f463576704203",
        allow_patterns=["*.json", "*.txt", "*.jinja"],
    )

    def init_qwen(model_id, dtype):
        c = AutoConfig.from_pretrained(p)
        # Build the non-persistent rotary tables on a real device as well.
        return Qwen3VLForConditionalGeneration._from_config(
            c, dtype=dtype, attn_implementation=attention
        ), AutoProcessor.from_pretrained(p)

    lawam.load_qwen3vl = init_qwen
    cfg = lm.LaWAMConfig.from_pretrained("ckpt")
    cfg.device = "cpu"
    cfg.base_vlm = str(p)
    model = lm.LaWAMPolicy(cfg).eval()
    if weights:
        model.load_state_dict(load_file("ckpt/model.safetensors"), strict=True)
    steps = json.loads(Path("ckpt/policy_preprocessor.json").read_text())["steps"]
    opts = next(s["config"] for s in steps if s["registry_name"] == "lawam_qwen_inputs")
    opts["model_id"] = str(p)
    proc = LaWAMQwenInputsProcessorStep(**opts)
    prepare = LaWAMPrepareBatchProcessorStep(
        **next(
            s["config"] for s in steps if s["registry_name"] == "lawam_prepare_batch"
        )
    )
    torch.manual_seed(123)
    cams = {
        k: torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
        for k, img in zip(
            ("observation.images.image", "observation.images.image2"),
            fixture_obs().images,
        )
    }
    tr = proc(
        {
            TransitionKey.OBSERVATION: cams,
            TransitionKey.COMPLEMENTARY_DATA: {"task": ["Pick up the cube."]},
        }
    )
    batch = prepare(tr)[TransitionKey.COMPLEMENTARY_DATA]
    return model, batch
