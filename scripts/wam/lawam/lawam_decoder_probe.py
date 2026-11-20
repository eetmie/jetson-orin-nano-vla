"""Local LIBERO LaWAM component probe. Run fetch, then build in a GPU container."""

import argparse
import json
import struct
import urllib.request
from pathlib import Path

REVISION = "f3d7b131bf612127a6e384ea84f563bc23564832"
URL = f"https://huggingface.co/jialei02/lawam-libero-sft-lerobot/resolve/{REVISION}/model.safetensors"
NAMES = ["layer1.W", "layer1.b", "layer2.W", "layer2.b"]


def read_range(start, length):
    end = start + length - 1
    req = urllib.request.Request(
        f"{URL}?slice={start}-{end}", headers={"Range": f"bytes={start}-{end}"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        if response.status != 206 or not response.headers["Content-Range"].startswith(
            f"bytes {start}-{end}/"
        ):
            raise ValueError("Unexpected checkpoint range")
        data = response.read(length)
        if len(data) != length:
            raise EOFError("Incomplete checkpoint range")
        return data


def fetch(folder):
    (size,) = struct.unpack("<Q", read_range(0, 8))
    index = json.loads(read_range(8, size))
    for name in NAMES:
        tensor = index["model.policy_backend.flow.action_decoder." + name]
        start, end = tensor["data_offsets"]
        assert tensor["dtype"] == "F32" and tensor["shape"][0] == 32
        stride = (end - start) // 32
        (folder / name).write_bytes(read_range(8 + size + start + 25 * stride, stride))


def build(folder):
    import numpy as np
    import tensorrt as trt
    import torch
    from torch import nn

    shapes = [(1024, 1024), (1024,), (1024, 32), (32,)]
    weights = [
        torch.from_numpy(np.fromfile(folder / name, dtype="<f4").reshape(shape))
        for name, shape in zip(NAMES, shapes)
    ]

    class Decoder(nn.Module):
        def forward(self, x):
            w1, b1, w2, b2 = weights
            return torch.relu(x @ w1 + b1) @ w2 + b2

    torch.manual_seed(2026)
    x = torch.randn(1, 50, 1024)
    reference = Decoder()(x)
    weights[:] = [w.half() for w in weights]
    graph = folder / "decoder.onnx"
    torch.onnx.export(
        Decoder(),
        x.half(),
        str(graph),
        dynamo=False,
        opset_version=17,
        input_names=["hidden"],
        output_names=["velocity"],
    )
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, log)
    if not parser.parse_from_file(str(graph)):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    config.builder_optimization_level = 1
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Build failed")
    (folder / "decoder.plan").write_bytes(bytes(plan))
    runtime = trt.Runtime(log)
    engine = runtime.deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()
    hidden = x.cuda().half()
    velocity = torch.empty_like(reference, device="cuda", dtype=torch.float16)
    ctx.set_tensor_address("hidden", hidden.data_ptr())
    ctx.set_tensor_address("velocity", velocity.data_ptr())
    assert ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    actual = velocity.float().cpu()
    cosine = torch.nn.functional.cosine_similarity(
        reference.flatten(), actual.flatten(), dim=0
    ).item()
    print(
        json.dumps(
            dict(
                cosine=cosine,
                max_abs=(actual - reference).abs().max().item(),
                plan_bytes=len(bytes(plan)),
                cgroup_peak_bytes=Path("/sys/fs/cgroup/memory.peak")
                .read_text()
                .strip(),
            )
        )
    )
    assert torch.isfinite(actual).all() and cosine >= 0.99999


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fetch", "build"])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    (fetch if args.stage == "fetch" else build)(args.folder)
