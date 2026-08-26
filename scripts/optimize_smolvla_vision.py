#!/usr/bin/env python3
"""Build a validated SmolVLA bundle without redundant vision NaN guards.

Recent PyTorch exports add ``IsNaN -> Where`` after twelve attention softmaxes.
The guards block a faster TensorRT vision graph. This tool is fail-closed: it matches
only the known pattern, never overwrites a destination, validates the rewritten ONNX,
and requires exact CPU ORT output equality on five deterministic inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


VISION_GRAPH = "smolvlm_vision.onnx"
EXPECTED_GUARDS = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _remove_nan_guards(model: Any) -> int:
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    removed_ids: set[int] = set()
    rewired = 0
    for matmul in model.graph.node:
        if matmul.op_type != "MatMul":
            continue
        for input_index, name in enumerate(matmul.input):
            where = producers.get(name)
            if where is None or where.op_type != "Where" or len(where.input) != 3:
                continue
            isnan = producers.get(where.input[0])
            where_users = consumers.get(where.output[0], [])
            isnan_users = consumers.get(isnan.output[0], []) if isnan else []
            if (
                isnan is None
                or isnan.op_type != "IsNaN"
                or list(isnan.input) != [where.input[2]]
                or len(where_users) != 1
                or where_users[0] is not matmul
                or len(isnan_users) != 1
                or isnan_users[0] is not where
            ):
                continue
            matmul.input[input_index] = where.input[2]
            removed_ids.update((id(where), id(isnan)))
            rewired += 1

    if rewired != EXPECTED_GUARDS or len(removed_ids) != EXPECTED_GUARDS * 2:
        raise RuntimeError(
            f"expected {EXPECTED_GUARDS} isolated IsNaN/Where guards, found "
            f"{rewired}; refusing to rewrite an unfamiliar graph"
        )
    kept = [node for node in model.graph.node if id(node) not in removed_ids]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return rewired


def _validate_outputs(source: Path, candidate: Path) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    reference = ort.InferenceSession(
        str(source), options, providers=["CPUExecutionProvider"]
    )
    optimized = ort.InferenceSession(
        str(candidate), options, providers=["CPUExecutionProvider"]
    )
    ref_inputs = reference.get_inputs()
    opt_inputs = optimized.get_inputs()
    if len(ref_inputs) != 1 or len(opt_inputs) != 1:
        raise RuntimeError("vision graph must have exactly one input")
    ref_input, opt_input = ref_inputs[0], opt_inputs[0]
    if (
        ref_input.name != opt_input.name
        or ref_input.type != "tensor(float)"
        or opt_input.type != "tensor(float)"
        or ref_input.shape != opt_input.shape
        or len(ref_input.shape) != 4
        or not all(isinstance(dim, int) and dim > 0 for dim in ref_input.shape)
    ):
        raise RuntimeError("source and candidate vision interfaces do not match")

    shape = tuple(ref_input.shape)
    rng = np.random.default_rng(20260826)
    ramp = np.linspace(-1.0, 1.0, shape[-1], dtype=np.float32)
    cases = (
        ("zeros", np.zeros(shape, dtype=np.float32)),
        ("half", np.full(shape, 0.5, dtype=np.float32)),
        ("uniform", rng.uniform(-1.0, 1.0, shape).astype(np.float32)),
        ("normal", rng.normal(0.0, 1.0, shape).astype(np.float32)),
        ("gradient", np.broadcast_to(ramp.reshape(1, 1, 1, -1), shape).copy()),
    )
    worst = 0.0
    for case_name, value in cases:
        expected = reference.run(None, {ref_input.name: value})
        actual = optimized.run(None, {opt_input.name: value})
        if len(expected) != len(actual):
            raise RuntimeError(f"{case_name}: output counts differ")
        for output_index, (left, right) in enumerate(zip(expected, actual)):
            if not np.isfinite(right).all():
                raise RuntimeError(f"{case_name}: output {output_index} is non-finite")
            difference = float(
                np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
            )
            worst = max(worst, difference)
            if difference != 0.0:
                raise RuntimeError(
                    f"{case_name}: output {output_index} differs by {difference:.9g}"
                )
    return {
        "provider": "CPUExecutionProvider",
        "onnxruntime": ort.__version__,
        "deterministic_cases": len(cases),
        "worst_max_abs": worst,
    }


def _write_manifest(bundle: Path) -> int:
    lines = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        lines.append(f"{_sha256(path)}  {path.relative_to(bundle)}")
    (bundle / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return len(lines)


def optimize_bundle(source: Path, destination: Path) -> dict[str, Any]:
    import onnx

    source = source.resolve()
    destination = destination.resolve()
    source_graph = source / VISION_GRAPH
    if not source.is_dir() or not source_graph.is_file():
        raise FileNotFoundError(f"missing split bundle or {VISION_GRAPH}: {source}")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    work = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )
    try:
        shutil.copytree(source, work, dirs_exist_ok=True)
        source_sha256 = _sha256(source_graph)

        # Check loaded weights, but serialize the wire model so external offsets and
        # inline shape constants retain the exporter's original layout.
        checked = onnx.load(str(source_graph), load_external_data=True)
        wire = onnx.load(str(source_graph), load_external_data=False)
        checked_count = _remove_nan_guards(checked)
        wire_count = _remove_nan_guards(wire)
        if checked_count != wire_count:
            raise RuntimeError("loaded and wire models produced different rewrite counts")
        onnx.checker.check_model(checked)

        candidate_graph = work / VISION_GRAPH
        temporary_graph = work / f".{VISION_GRAPH}.tmp"
        temporary_graph.write_bytes(wire.SerializeToString())
        onnx.checker.check_model(str(temporary_graph))
        os.replace(temporary_graph, candidate_graph)
        onnx.checker.check_model(str(candidate_graph))
        validation = _validate_outputs(source_graph, candidate_graph)

        info_path = work / "export_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"missing export metadata: {info_path}")
        info = json.loads(info_path.read_text())
        optimization = {
            "name": "remove_redundant_vision_nan_guards",
            "graph": VISION_GRAPH,
            "source_sha256": source_sha256,
            "rewired_matmuls": checked_count,
            "removed_nodes": {"IsNaN": checked_count, "Where": checked_count},
            "validation": validation,
        }
        existing = info.get("post_export_optimizations", [])
        if not isinstance(existing, list):
            raise RuntimeError("post_export_optimizations must be a list when present")
        info["post_export_optimizations"] = [*existing, optimization]
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        manifest_files = _write_manifest(work)
        result = {
            "source": str(source),
            "destination": str(destination),
            "vision_sha256": _sha256(candidate_graph),
            "manifest_files": manifest_files,
            "optimization": optimization,
        }
        work.rename(destination)
        return result
    except BaseException:
        if work.exists():
            shutil.rmtree(work)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original split-export bundle")
    parser.add_argument("destination", type=Path, help="new optimized bundle")
    args = parser.parse_args()
    print(json.dumps(optimize_bundle(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
