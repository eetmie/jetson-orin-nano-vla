import sys
import torch
from reference import load_and_prepare
from torch import nn

# --fp32: an FP32 graph for vla_common.fp16_mixed instead of the stock BF16 autocast.
fp32 = "--fp32" in sys.argv


class Dino(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, primary):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32):
            return self.encoder.encode(primary, n=-2)[:, 0].float()


with torch.inference_mode():
    model, b = load_and_prepare()
    net = Dino(model.model.policy_backend.lam.vision_encoder.cuda().eval())
    x = b["primary_image"].cuda()
    torch.onnx.export(
        net,
        (x,),
        "dino-fp32.onnx" if fp32 else "dino-native.onnx",
        dynamo=True,
        input_names=["primary"],
        output_names=["ht"],
        opset_version=18,
        external_data=False,
    )
