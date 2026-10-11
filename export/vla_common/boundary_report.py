#!/usr/bin/env python3
"""Which FP32 engine-boundary tensors of a mixed-FP16 bundle could cross in FP16 exactly.

A graph input is FP16-read when every consumer is Cast(to FP16). If that holds on every
graph that reads a boundary, rounding once at the producer is the rounding each consumer
already does: passing it in FP16 changes no value and halves its buffer and traffic.

An output whose ONNX producer is Cast(to FP32) of an FP16 tensor is NOT enough on its own:
TensorRT fuses that FP16 op into an FP32-accumulating GEMM epilogue and emits more than
FP16 precision. Measured on GR00T N1.6: FP16 hidden states between the LLM/DiT/vision
chunks doubled the fixture error (0.060 -> 0.130 %). The producer column is reported for
information only.

    python -m vla_common.boundary_report <bundle dir> [--link out=in ...]

--link names a host hop: an output the runtime copies into another graph's input (GR00T
scatters `vision_tokens` into the LLM's `h`), so both count as one boundary.
"""

from __future__ import annotations

import argparse
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


def _link(name: str, links: dict[str, str] | None = None) -> str:
    """The name a split graph's output takes as the next graph's input."""
    if links and name in links:
        return links[name]
    if name == "hidden_out":
        return "hidden_in"
    for suffix in ("_out", "_next"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def parse_links(items: list[str] | None) -> dict[str, str]:
    return dict(x.split("=", 1) for x in items or [])


def boundaries(root: Path, links: dict[str, str] | None = None):
    """{boundary: [(graph, ok)]} for producers and consumers of every FP32 boundary."""
    produced, consumed = defaultdict(list), defaultdict(list)
    for f in sorted(root.glob("*.onnx")):
        outs, ins = scan(f)
        for k, ok in outs.items():
            produced[_link(k, links)].append((f.stem, ok))
        for k, ok in ins.items():
            consumed[k].append((f.stem, ok))
    return produced, consumed


def exact(name: str, produced, consumed) -> str | None:
    """'reads' when every consumer reads FP16 (see the module docstring), else None."""
    if all(ok for _, ok in consumed[name]):
        return "reads"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--link", action="append", metavar="OUT=IN")
    a = ap.parse_args()
    produced, consumed = boundaries(a.bundle, parse_links(a.link))
    names = sorted(set(produced) & set(consumed))
    why = {k: exact(k, produced, consumed) for k in names}
    print(f"{a.bundle.name}: {len(names)} FP32 tensors cross engines, "
          f"{sum(v is not None for v in why.values())} would be exact in FP16")
    for k in names:
        tag = f"exact ({why[k]})" if why[k] else "NOT exact"
        print(f"  {tag:16s} {k:28s} from {[s for s, _ in produced[k]]} -> {len(consumed[k])} graph(s)"
              + ("" if why[k] else f"  producers {produced[k]} consumers {consumed[k][:3]}"))


if __name__ == "__main__":
    main()
