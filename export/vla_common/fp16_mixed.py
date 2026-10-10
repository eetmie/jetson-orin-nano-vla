#!/usr/bin/env python3
"""Mixed-FP16 graphs for a strongly typed TensorRT build, with every RMSNorm kept FP32.

GR00T's pass (groot/export_split_onnx.py) generalized to graphs whose node names do not
say which nodes are a norm: an RMSNorm is found by its shape instead,
Pow -> ReduceMean -> Add(eps) -> Sqrt -> Reciprocal|Div -> Mul -> Mul(weight), and the
whole chain is kept FP32, as stock PyTorch upcasts it. LayerNormalization and Softmax
stay FP32 too. A strongly typed engine computes exactly what the ONNX says, so this is
where the precision is decided, not at build time.

Two fixes to ORT's float16 converter are applied (see GR00T's exporter for the measured
failures): its FP32->FP16->FP32 Cast round trips between two FP32-kept nodes are removed,
and Casts it inserts twice are dropped.

    python -m vla_common.fp16_mixed --bundle <dir> --graphs a.onnx b.onnx
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import onnx

from .bundle.manifest import MANIFEST_NAME, write_manifest
from .fp16_weights import FP16_SENSITIVE_OPS

_RMS_TAIL = (("Add",), ("Sqrt",), ("Reciprocal", "Div"), ("Mul",), ("Mul",))


def rmsnorm_nodes(m: onnx.ModelProto) -> list[str]:
    """Names of every node in an RMSNorm chain; unnamed ones are given names."""
    prod = {o: n for n in m.graph.node for o in n.output}
    cons: dict[str, list] = {}
    for n in m.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    keep = set()
    for n in m.graph.node:
        p = prod.get(n.input[0]) if n.op_type == "ReduceMean" else None
        if p is None or p.op_type != "Pow":
            continue
        chain, cur = [p, n], n
        for want in _RMS_TAIL:
            nxt = [c for c in cons.get(cur.output[0], []) if c.op_type in want]
            if not nxt:
                break
            cur = nxt[0]
            chain.append(cur)
        keep.update(id(c) for c in chain)
    for i, n in enumerate(m.graph.node):
        if id(n) in keep and not n.name:
            n.name = f"rmsnorm_fp32_{i}"
    return [n.name for n in m.graph.node if id(n) in keep]


def strip_fp16_roundtrips(m: onnx.ModelProto) -> int:
    """Remove Cast(FP32->FP16) -> Cast(->FP32) pairs; consumers read the FP32 source."""
    typed = onnx.shape_inference.infer_shapes(m)
    types = {v.name: v.type.tensor_type.elem_type
             for v in list(typed.graph.value_info) + list(typed.graph.input)
             + list(typed.graph.output)}
    types.update({i.name: i.data_type for i in m.graph.initializer})
    prod = {o: n for n in m.graph.node for o in n.output}
    outputs = {o.name for o in m.graph.output}

    def to(n):
        return next(a.i for a in n.attribute if a.name == "to")

    F32, F16 = onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16
    rename = {}
    for n in m.graph.node:
        if n.op_type == "Cast" and to(n) == F32 and n.output[0] not in outputs:
            p = prod.get(n.input[0])
            if (p is not None and p.op_type == "Cast" and to(p) == F16
                    and types.get(p.input[0]) == F32):
                rename[n.output[0]] = p.input[0]
    for n in m.graph.node:
        for i, x in enumerate(n.input):
            if x in rename:
                n.input[i] = rename[x]
    while True:                      # drop nodes nothing reads any more
        used = {x for n in m.graph.node for x in n.input} | outputs
        keep = [n for n in m.graph.node if any(o in used for o in n.output)]
        if len(keep) == len(m.graph.node):
            break
        del m.graph.node[:]
        m.graph.node.extend(keep)
    return len(rename)


def dedupe_casts(m: onnx.ModelProto) -> int:
    """Drop exact repeats of a node the converter inserted twice (same outputs)."""
    seen, keep = {}, []
    for n in m.graph.node:
        key = tuple(n.output)
        if key in seen:
            first = seen[key]
            assert (first.op_type, list(first.input), first.attribute) == \
                (n.op_type, list(n.input), n.attribute), f"{key} defined twice, differently"
            continue
        seen[key] = n
        keep.append(n)
    dropped = len(m.graph.node) - len(keep)
    if dropped:
        del m.graph.node[:]
        m.graph.node.extend(keep)
    return dropped


def convert(src: Path, dst: Path, half_io: str | None = None) -> dict:
    from onnxruntime.transformers import float16
    from onnxruntime.transformers.onnx_model import OnnxModel

    m = onnx.load(str(src))
    # The tracer's value_info still says FP32 where the converter makes FP16, and shape
    # inference in the round-trip pass rejects the disagreement.
    del m.graph.value_info[:]
    norms = rmsnorm_nodes(m)
    keep = True
    if half_io:
        # I/O matching half_io is FP16 at the engine boundary; everything else stays FP32.
        pattern = re.compile(half_io)
        names = [v.name for v in list(m.graph.input) + list(m.graph.output)]
        keep = [n for n in names if not pattern.fullmatch(n)]
    out = float16.convert_float_to_float16(
        m, keep_io_types=keep, node_block_list=norms,
        op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + list(FP16_SENSITIVE_OPS))
    stripped = strip_fp16_roundtrips(out)
    # save_model_to_file sorts topologically (keep_io_types appends input Casts last).
    om = OnnxModel(out)
    om.topological_sort()
    deduped = dedupe_casts(om.model)
    tmp = dst.with_name(dst.name + ".tmp")
    om.save_model_to_file(str(tmp), use_external_data_format=False)
    onnx.checker.check_model(str(tmp))
    tmp.replace(dst)
    return {"rmsnorm_nodes": len(norms), "roundtrips_removed": stripped,
            "duplicates_dropped": deduped}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--graphs", nargs="+", required=True,
                    help="converted in place; their external .data files are removed, and "
                         "bundle.json sizes and MANIFEST.sha256 are rewritten if present")
    ap.add_argument("--half-io", metavar="REGEX",
                    help="graph inputs/outputs whose whole name matches stay FP16 at the "
                         "boundary (e.g. a KV cache passed between two engines)")
    a = ap.parse_args()
    sizes = {}
    for g in a.graphs:
        src = a.bundle / g
        data = src.with_name(src.name + ".data")
        before = src.stat().st_size + (data.stat().st_size if data.exists() else 0)
        r = convert(src, src, a.half_io)
        data.unlink(missing_ok=True)
        sizes[g] = round(src.stat().st_size / 1e6, 1)
        print(f"  {g:30s} {before / 1e6:5.0f} MB -> {sizes[g]:5.0f} MB  {r}")
    meta = a.bundle / "bundle.json"
    if meta.exists():
        doc = json.loads(meta.read_text())
        for graph in doc.get("graphs", []):
            if isinstance(graph, dict) and graph.get("file") in sizes:
                graph["size_mb"] = sizes[graph["file"]]
                graph["precision"] = "mixed fp16"
        meta.write_text(json.dumps(doc, indent=2) + "\n")
    if (a.bundle / MANIFEST_NAME).exists():
        write_manifest(a.bundle)


if __name__ == "__main__":
    main()
