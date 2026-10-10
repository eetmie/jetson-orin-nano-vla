#!/usr/bin/env python3
"""Pass chosen FP32 engine-boundary tensors of a mixed-FP16 bundle in FP16, exactly.

Only tensors that today round-trip FP16 -> FP32 -> FP16 (see boundary_report) are
touched: the producer's Cast(to FP32) and the consumers' Cast(to FP16) become FP16
identities and the graph I/O is retyped, so no value changes. Anything matching
--names that is not provably exact is refused.

    python -m vla_common.half_boundary --bundle <dir> --names 'kv_[0-9]+|mod_[0-9]+'
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import onnx

from .boundary_report import F16, F32, _link, _to, scan
from .bundle.manifest import MANIFEST_NAME, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--names", required=True, help="regex over boundary names (an output's "
                    "linked name: hidden_out -> hidden_in, h_out -> h)")
    a = ap.parse_args()
    pattern = re.compile(a.names)
    graphs = sorted(a.bundle.glob("*.onnx"))
    produced, consumed = defaultdict(list), defaultdict(list)
    for f in graphs:
        outs, ins = scan(f)
        for k, ok in outs.items():
            produced[_link(k)].append(ok)
        for k, ok in ins.items():
            consumed[k].append(ok)
    chosen = sorted(k for k in set(produced) & set(consumed) if pattern.fullmatch(k))
    refused = [k for k in chosen if not all(produced[k] + consumed[k])]
    if refused:
        raise SystemExit(f"not exact in FP16, refusing: {refused}")
    if not chosen:
        raise SystemExit("nothing matches")
    changed = {}
    for f in graphs:
        m = onnx.load(str(f))
        g = m.graph
        prod = {o: n for n in g.node for o in n.output}
        edits = 0
        for o in g.output:
            if _link(o.name) in chosen and o.type.tensor_type.elem_type == F32:
                p = prod[o.name]
                assert p.op_type == "Cast" and _to(p) == F32
                next(x for x in p.attribute if x.name == "to").i = F16
                o.type.tensor_type.elem_type = F16
                edits += 1
        for i in g.input:
            if i.name in chosen and i.type.tensor_type.elem_type == F32:
                for n in g.node:
                    if i.name in n.input:
                        assert n.op_type == "Cast" and _to(n) == F16
                i.type.tensor_type.elem_type = F16
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
        meta.write_text(json.dumps(doc, indent=2) + "\n")
    if (a.bundle / MANIFEST_NAME).exists():
        write_manifest(a.bundle)
    print(f"FP16 boundary for {len(chosen)} tensors in {len(changed)} graphs: {changed}")


if __name__ == "__main__":
    main()
