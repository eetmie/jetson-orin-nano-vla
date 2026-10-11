#!/usr/bin/env python3
"""Pass chosen FP32 engine-boundary tensors of a mixed-FP16 bundle in FP16, exactly.

Only boundaries boundary_report proves exact are touched: every consumer reads FP16.
Producers round to FP16 at the output (their existing Cast(to FP32) becomes an FP16
identity, or a Cast(to FP16) is appended) and the consumers read the FP16 input directly.
No value changes. Anything matching --names that is not provably exact is refused.

    python -m vla_common.half_boundary --bundle <dir> --names 'kv_[0-9]+|mod_[0-9]+'
                                       [--link vision_tokens=h]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import onnx
from onnx import helper

from .boundary_report import F16, F32, _link, _to, boundaries, exact, parse_links
from .bundle.manifest import MANIFEST_NAME, write_manifest


def _half_output(g, o) -> None:
    prod = next(n for n in g.node if o.name in n.output)
    inside = [n for n in g.node if o.name in n.input]
    if prod.op_type == "Cast" and _to(prod) == F32 and not inside:
        next(x for x in prod.attribute if x.name == "to").i = F16
    else:
        wide = o.name + "_fp32"
        for n in [prod, *inside]:
            for lst in (n.output, n.input):
                for j, x in enumerate(lst):
                    if x == o.name:
                        lst[j] = wide
        g.node.append(helper.make_node("Cast", [wide], [o.name], to=F16,
                                       name=o.name + "_to_fp16"))
    o.type.tensor_type.elem_type = F16


def _half_input(g, i) -> None:
    wide_readers = [n for n in g.node if i.name in n.input
                    and not (n.op_type == "Cast" and _to(n) == F16)]
    if wide_readers:
        wide = i.name + "_fp32"
        for n in wide_readers:
            for j, x in enumerate(n.input):
                if x == i.name:
                    n.input[j] = wide
        g.node.insert(0, helper.make_node("Cast", [i.name], [wide], to=F32,
                                          name=i.name + "_to_fp32"))
    i.type.tensor_type.elem_type = F16


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--names", required=True, help="regex over boundary names (an output's "
                    "linked name: hidden_out -> hidden_in, h_out -> h)")
    ap.add_argument("--link", action="append", metavar="OUT=IN",
                    help="an output the runtime copies into another graph's input")
    a = ap.parse_args()
    links = parse_links(a.link)
    pattern = re.compile(a.names)
    graphs = sorted(a.bundle.glob("*.onnx"))
    produced, consumed = boundaries(a.bundle, links)
    chosen = sorted(k for k in set(produced) & set(consumed) if pattern.fullmatch(k))
    refused = [k for k in chosen if exact(k, produced, consumed) is None]
    if refused:
        raise SystemExit(f"not exact in FP16, refusing: {refused}")
    if not chosen:
        raise SystemExit("nothing matches")
    changed = {}
    for f in graphs:
        m = onnx.load(str(f))
        g = m.graph
        edits = 0
        for o in g.output:
            if _link(o.name, links) in chosen and o.type.tensor_type.elem_type == F32:
                _half_output(g, o)
                edits += 1
        for i in g.input:
            if i.name in chosen and i.type.tensor_type.elem_type == F32:
                _half_input(g, i)
                edits += 1
        if edits:
            onnx.checker.check_model(m)
            tmp = f.with_name(f.name + ".tmp")
            onnx.save(m, str(tmp))
            tmp.replace(f)
            changed[f.name] = edits
    meta = a.bundle / "bundle.json"
    if meta.exists():
        doc = json.loads(meta.read_text())
        doc["half_boundary"] = sorted(set(doc.get("half_boundary", [])) | set(chosen))
        if links:
            doc["half_boundary_links"] = {**doc.get("half_boundary_links", {}), **links}
        meta.write_text(json.dumps(doc, indent=2) + "\n")
    if (a.bundle / MANIFEST_NAME).exists():
        write_manifest(a.bundle)
    print(f"FP16 boundary for {len(chosen)} tensors in {len(changed)} graphs: {changed}")


if __name__ == "__main__":
    main()
