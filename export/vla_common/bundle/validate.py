# Vendored from the author's fine-tuning pipeline: vla-onnx/common/vla_common/bundle/validate.py @ 3f7793d.
"""Reject a malformed graph at export time rather than at engine-build time."""

from __future__ import annotations

from pathlib import Path


def validate_onnx(path: str | Path, stem: str | None = None) -> None:
    """Raise if `path` has a node with an empty REQUIRED input, then run the checker.

    The TorchScript exporter can silently emit nodes whose required inputs are empty
    strings; the file writes fine and only fails minutes later when ORT tries to load
    it ("input 0 is marked single but has an empty string"). Checking here keeps that
    failure next to the code that caused it.

    An empty string is only a defect in a REQUIRED position. ONNX also uses "" as the
    legal way to omit a trailing optional input, which is exactly what the tracer
    emits for the DaViT window-attention `Pad` (`constant_value` omitted, because the
    pads are all zero at 224x224 — every feature map divides by the window size).
    Flagging those rejected a graph that `onnx.checker` passes and that ORT loads
    happily, so the op schema decides which positions matter rather than a blanket
    rule. When the schema is unknown the blanket rule is restored, so an unrecognised
    op fails closed rather than slipping through.
    """
    import onnx
    from onnx import defs

    path = Path(path)
    stem = stem or path.name
    model = onnx.load(str(path), load_external_data=False)
    opset = {o.domain: o.version for o in model.opset_import}

    def required_positions(op_type: str, domain: str) -> set[int] | None:
        """Indices whose input must be present, or None when the schema is unknown."""
        try:
            schema = defs.get_schema(
                op_type, opset.get(domain, opset.get("", 17)), domain
            )
        except Exception:
            return None
        return {
            i
            for i, inp in enumerate(schema.inputs)
            if inp.option != defs.OpSchema.FormalParameterOption.Optional
        }

    dangling = []
    for n in model.graph.node:
        req = required_positions(n.op_type, n.domain)
        for i, name in enumerate(n.input):
            if name != "":
                continue
            if req is None or i in req:
                dangling.append((n.op_type, n.name, i))
    if dangling:
        raise RuntimeError(
            f"{stem}: {len(dangling)} node(s) exported with an empty REQUIRED input, "
            f"e.g. {dangling[:3]} — the graph is unloadable. Usually a traced construct "
            f"the exporter mishandled (in-place indexed assignment, expand(-1, ...))."
        )
    onnx.checker.check_model(model)
