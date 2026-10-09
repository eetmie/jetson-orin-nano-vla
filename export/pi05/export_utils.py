# Vendored from the author's pi0.5 Orin Nano prototype (spark-projects): pi05-spark-inference/prototype/pi05_fp16_full_20261010/export_utils.py.
# Runs in the openpi container (export/pi05/README.md); paths are that stage layout's.
"""ONNX accumulation controls and bounded per-layer refit weight bundles."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def accumulation(graph):
    inferred = shape_inference.infer_shapes(graph)
    types = {v.name: v.type.tensor_type.elem_type for v in (
        list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info))}
    types.update({v.name: v.data_type for v in graph.graph.initializer})
    nodes, changed = [], 0
    for i, node in enumerate(graph.graph.node):
        if node.op_type in ('MatMul', 'Conv') and all(types.get(x) == TensorProto.FLOAT16 for x in node.input):
            operands = []
            for j, operand in enumerate(node.input):
                output = f'fp32_accum_input_{i}_{j}'
                nodes.append(helper.make_node('Cast', [operand], [output], to=TensorProto.FLOAT))
                operands.append(output)
            original_output = node.output[0]
            del node.input[:]
            node.input.extend(operands)
            node.output[0] = f'fp32_accum_output_{i}'
            nodes.append(node)
            nodes.append(helper.make_node('Cast', [node.output[0]], [original_output], to=TensorProto.FLOAT16))
            changed += 1
        else:
            nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph)
    return changed


def structure_hash(graph):
    copy = onnx.ModelProto()
    copy.CopyFrom(graph)
    for tensor in copy.graph.initializer:
        if tensor.data_type in (TensorProto.FLOAT16, TensorProto.FLOAT):
            for field in ('raw_data', 'float_data', 'double_data', 'int32_data', 'int64_data'):
                tensor.ClearField(field)
    return hashlib.sha256(copy.SerializeToString()).hexdigest()


def pack(graph, kind, label, template=False):
    directory = ROOT / 'weights' / label
    directory.mkdir(parents=True, exist_ok=True)
    descriptors = []
    for i, tensor in enumerate(graph.graph.initializer):
        if tensor.data_type not in (TensorProto.FLOAT16, TensorProto.FLOAT):
            continue
        array = np.array(numpy_helper.to_array(tensor), copy=True, order='C')
        name = f'{i:03d}.npy'
        np.save(directory / name, array)
        descriptors.append({'name': tensor.name, 'file': name, 'dtype': str(array.dtype),
                            'shape': list(array.shape), 'bytes': array.nbytes,
                            'sha256': sha(directory / name)})
    record = {'kind': kind, 'label': label, 'structure_sha256': structure_hash(graph),
              'weights': descriptors, 'initializer_bytes': sum(x['bytes'] for x in descriptors)}
    (directory / 'index.json').write_text(json.dumps(record, indent=2))
    if template:
        (ROOT / 'templates').mkdir(exist_ok=True)
        onnx.save(graph, ROOT / 'templates' / (kind + '.onnx'))
        (ROOT / 'templates' / (kind + '.json')).write_text(json.dumps(record, indent=2))
    else:
        expected = json.loads((ROOT / 'templates' / (kind + '.json')).read_text())
        assert expected['structure_sha256'] == record['structure_sha256'], (
            'Different graph structure requires its own template', kind, label)
    print('Packed', label, 'raw bytes', record['initializer_bytes'], flush=True)
    return record


if __name__ == '__main__':
    # Fast refit pilot against the already validated first language graph.
    source = ROOT.parent / 'orin_initial_20261010_001/prefix_accum32.onnx'
    graph = onnx.load(source)
    pack(graph, 'prefix', 'prefix_00', template=True)
