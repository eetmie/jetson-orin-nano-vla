import torch
from reference import load_and_prepare
from torch import nn


class Dino(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, primary):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            return self.encoder.encode(primary, n=-2)[:, 0].float()


with torch.inference_mode():
    model, b = load_and_prepare()
    net = Dino(model.model.policy_backend.lam.vision_encoder.cuda().eval())
    x = b["primary_image"].cuda()
    torch.onnx.export(
        net,
        (x,),
        "dino-native.onnx",
        dynamo=True,
        input_names=["primary"],
        output_names=["ht"],
        opset_version=18,
        external_data=False,
    )
