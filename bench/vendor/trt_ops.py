# ─────────────────────────────────────────────────────────────────────────────
# Written for this repo (not vendored).
# ─────────────────────────────────────────────────────────────────────────────
"""Tiny elementwise TensorRT engines for the glue between a policy's graphs.

The host loops do a little numpy between engines: X-VLA's interpolation, an Euler
update, SmolVLA's SiLU. On a device-resident chain that math has to run on the GPU too,
and the board's venv has neither torch nor onnx, so these are built straight from the
TensorRT network API, FP32 and strongly typed, and cached beside the model's engines.

    lerp   out = x1 * w + a * (1 - w)     w is a [1,1,1] input, one buffer per step
    axpy   out = a + c * b                c is baked in
    silu   out = x * sigmoid(x)
    siglip_u8  out[1,3,H,W] = lut[x[H,W,3]]   SigLIP's [-1,1] scaling of a uint8 image
                                              as a table gather, so it is exact
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _key(op: str, shape: tuple, c: float | None) -> str:
    spec = json.dumps({"op": op, "shape": list(shape), "c": c}, sort_keys=True)
    return f"op_{op}_{hashlib.sha256(spec.encode()).hexdigest()[:12]}"


def build_op(op: str, shape: list[int], c: float | None, path: str) -> None:
    import numpy as np
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    f32 = trt.float32
    E = trt.ElementWiseOperation
    if op == "lerp":
        x1 = net.add_input("x1", f32, shape)
        a = net.add_input("a", f32, shape)
        w = net.add_input("w", f32, [1] * len(shape))
        one = net.add_constant([1] * len(shape), np.ones([1] * len(shape), np.float32))
        omw = net.add_elementwise(one.get_output(0), w, E.SUB).get_output(0)
        p = net.add_elementwise(x1, w, E.PROD).get_output(0)
        q = net.add_elementwise(a, omw, E.PROD).get_output(0)
        out = net.add_elementwise(p, q, E.SUM).get_output(0)
    elif op == "axpy":
        a = net.add_input("a", f32, shape)
        b = net.add_input("b", f32, shape)
        k = net.add_constant([1] * len(shape),
                             np.full([1] * len(shape), c, np.float32))
        cb = net.add_elementwise(b, k.get_output(0), E.PROD).get_output(0)
        out = net.add_elementwise(a, cb, E.SUM).get_output(0)
    elif op == "silu":
        x = net.add_input("x", f32, shape)
        sg = net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
        out = net.add_elementwise(x, sg, E.PROD).get_output(0)
    elif op == "siglip_u8":
        from .smolvla_split import siglip_lut

        h, w, ch = shape
        x = net.add_input("x", trt.uint8, shape)
        idx = net.add_cast(x, trt.int32).get_output(0)
        table = np.ascontiguousarray(siglip_lut())      # must outlive the build
        lut = net.add_constant(trt.Dims([256]), trt.Weights(table)).get_output(0)
        g = net.add_gather(lut, idx, 0).get_output(0)            # [H,W,3] fp32
        sh = net.add_shuffle(g)
        sh.first_transpose = trt.Permutation((2, 0, 1))
        sh.reshape_dims = (1, ch, h, w)
        out = sh.get_output(0)
    else:
        raise ValueError(op)
    out.name = "out"
    net.mark_output(out)
    cfg = builder.create_builder_config()
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise SystemExit(f"build failed: {op} {shape}")
    Path(path).write_bytes(blob)


class Ops:
    """Builds (in a subprocess, once) and loads the op engines a chain asks for."""

    def __init__(self, engines, cache_dir: str | Path):
        self.e = engines
        self.cache = Path(cache_dir).expanduser()
        self.contexts: dict[str, object] = {}

    def get(self, op: str, shape, c: float | None = None) -> str:
        """Engine name of an op, built and loaded on first use."""
        shape = tuple(int(x) for x in shape)
        name = _key(op, shape, c)
        if name in self.e.engines:
            return name
        path = self.cache / f"{name}.engine"
        if not path.exists():
            subprocess.run(
                [sys.executable, "-c",
                 "import sys, json; from bench.vendor.trt_ops import build_op; "
                 "build_op(sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4])",
                 op, json.dumps(list(shape)), json.dumps(c), str(path)],
                check=True, cwd=str(Path(__file__).resolve().parents[2]))
        self.e.add(name, path)
        return name
