#!/usr/bin/env python3
"""Make a mixed-FP16 graph's FP16 MatMuls accumulate in FP32, π0.5-style.

Each FP16 MatMul (both operands FP16) becomes Cast(FP32) -> MatMul -> Cast(FP16). The
operands are FP16 values, so every product is exact in FP32; TensorRT turns the pattern
into FP16-input, FP32-accumulate tensor-core kernels instead of picking FP16-accumulating
ones. Accuracy goes up; GEMMs TensorRT had run FP16-accumulating get slower.

    python -m vla_common.fp32_accumulate --bundle <dir> --graphs a.onnx b.onnx [--skip REGEX]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import onnx
from onnx import helper

from .bundle.manifest import MANIFEST_NAME, write_manifest

F32, F16 = onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16


def convert(path: Path, skip: re.Pattern | None) -> int:
    m = onnx.load(str(path))
    typed = onnx.shape_inference.infer_shapes(m)
    types = {v.name: v.type.tensor_type.elem_type
             for v in list(typed.graph.value_info) + list(typed.graph.input) + list(typed.graph.output)}
    types.update({i.name: i.data_type for i in m.graph.initializer})
    nodes, n = [], 0
    for node in m.graph.node:
        if (node.op_type == "MatMul" and all(types.get(i) == F16 for i in node.input)
                and not (skip and skip.search(node.name))):
            ins = []
            for k, x in enumerate(node.input):
                y = f"{node.name or x}_acc32_in{k}"
                nodes.append(helper.make_node("Cast", [x], [y], to=F32, name=y))
                ins.append(y)
            out32 = node.output[0] + "_acc32"
            nodes.append(helper.make_node("MatMul", ins, [out32], name=node.name))
            nodes.append(helper.make_node("Cast", [out32], [node.output[0]], to=F16,
                                          name=node.output[0] + "_acc32_out"))
            n += 1
        else:
            nodes.append(node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    tmp = path.with_name(path.name + ".tmp")
    onnx.save(m, str(tmp))
    tmp.replace(path)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--skip", help="regex over MatMul node names to leave alone")
    a = ap.parse_args()
    skip = re.compile(a.skip) if a.skip else None
    for g in a.graphs:
        print(f"  {g}: {convert(a.bundle / g, skip)} MatMuls now accumulate in FP32")
    if (a.bundle / MANIFEST_NAME).exists():
        write_manifest(a.bundle)


if __name__ == "__main__":
    main()
