#!/usr/bin/env python3
"""Which FP32 engine-boundary tensors of a mixed-FP16 bundle could cross in FP16 exactly.

A graph output qualifies when its producer is Cast(to FP32) of an FP16 tensor; a graph
input when every consumer is Cast(to FP16). A tensor that qualifies on every graph that
produces or consumes it round-trips FP16 -> FP32 -> FP16 today, so passing it in FP16
(fp16_mixed --half-io) changes no value and halves its buffer and traffic.

    python -m vla_common.boundary_report <bundle dir>
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import onnx

F32, F16 = onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16


def _to(n):
    return next((a.i for a in n.attribute if a.name == "to"), None)


def scan(path: Path):
    m = onnx.load(str(path), load_external_data=False)
    g = m.graph
    types = {v.name: v.type.tensor_type.elem_type for v in list(g.input) + list(g.output)}
    prod = {o: n for n in g.node for o in n.output}
    cons = defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    half_types = {}
    try:
        inferred = onnx.shape_inference.infer_shapes(m)
        half_types = {v.name: v.type.tensor_type.elem_type for v in inferred.graph.value_info}
    except Exception:
        pass
    outs, ins = {}, {}
    for o in g.output:
        if types[o.name] != F32:
            continue
        p = prod.get(o.name)
        outs[o.name] = bool(p is not None and p.op_type == "Cast" and _to(p) == F32
                            and half_types.get(p.input[0], None) == F16)
    for i in g.input:
        if types[i.name] != F32:
            continue
        c = cons.get(i.name, [])
        ins[i.name] = bool(c) and all(n.op_type == "Cast" and _to(n) == F16 for n in c)
    return outs, ins


def _link(name: str) -> str:
    """The name a split graph's output takes as the next graph's input."""
    if name == "hidden_out":
        return "hidden_in"
    for suffix in ("_out", "_next"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def main() -> None:
    root = Path(sys.argv[1])
    produced, consumed = defaultdict(list), defaultdict(list)
    for f in sorted(root.glob("*.onnx")):
        outs, ins = scan(f)
        for k, ok in outs.items():
            produced[_link(k)].append((f.stem, ok))
        for k, ok in ins.items():
            consumed[k].append((f.stem, ok))
    names = sorted(set(produced) & set(consumed))
    exact = [k for k in names if all(ok for _, ok in produced[k] + consumed[k])]
    print(f"{root.name}: {len(names)} FP32 tensors cross engines, {len(exact)} would be exact in FP16")
    for k in names:
        tag = "exact" if k in exact else "NOT exact"
        print(f"  {tag:9s} {k:28s} from {[s for s, _ in produced[k]]} -> {len(consumed[k])} graph(s)"
              + ("" if k in exact else f"  producers {produced[k]} consumers {consumed[k][:3]}"))


if __name__ == "__main__":
    main()
