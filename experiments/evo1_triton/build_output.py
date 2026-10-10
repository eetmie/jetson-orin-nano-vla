"""Rebuild EVO1's action_output engine with the pool GEMV as an FP32-accumulating plugin.

The pool layer is [1,44800] @ W[896,44800]^T + b, bandwidth-bound; TensorRT runs it on
an FP16-accumulating tensor-core tactic. Other engines are copied byte-identical from
--base-cache, a plain prebuilt cache of the same bundle.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import tensorrt as trt
import tensorrt.plugin as trtp
import plugin
from candidate import pin, sha256

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['baseline', 'gemv'], required=True)
p.add_argument('--bundle', type=Path, required=True)
p.add_argument('--base-cache', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
bundle, base, out = (x.expanduser().resolve() for x in (a.bundle, a.base_cache, a.out))
out.mkdir(parents=True, exist_ok=False)
for path in base.glob('*.engine'):
    if path.stem != 'action_output' and not path.name.startswith('op_'):
        shutil.copy2(path, out/path.name)
T = trt.LayerType
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
parser = trt.OnnxParser(net, logger)
assert parser.parse_from_file(str(bundle/'action_output.onnx')), [str(parser.get_error(i)) for i in range(parser.num_errors)]
layers = [net.get_layer(i) for i in range(net.num_layers)]
producers = {l.get_output(j).name: l for l in layers for j in range(l.num_outputs)}


def users(t):
    return [(l, j) for l in layers for j in range(l.num_inputs) if l.get_input(j) is not None and l.get_input(j).name == t.name]


def constant(t):
    l = producers[t.name]
    while l.type == T.SHUFFLE:
        l.__class__ = trt.IShuffleLayer
        l = producers[l.get_input(0).name]
    assert l.type == T.CONSTANT, l.name
    l.__class__ = trt.IConstantLayer
    return np.asarray(l.weights).reshape(tuple(l.shape))


keep, patch = [], None
if a.mode == 'gemv':
    mms = []
    for l in layers:
        if l.type == T.MATRIX_MULTIPLY:
            l.__class__ = trt.IMatrixMultiplyLayer
            if plugin.K in tuple(l.get_input(1).shape):
                mms.append(l)
    (mm,) = mms
    w = constant(mm.get_input(1))
    w = w if mm.op1 == trt.MatrixOperation.TRANSPOSE else w.T
    assert mm.op0 == trt.MatrixOperation.NONE and w.shape == (plugin.N, plugin.K) and w.dtype == np.float16, (w.shape, mm.op1)
    x = mm.get_input(0)
    assert tuple(x.shape) == (1, plugin.K) and x.dtype == trt.float16
    (add, slot), = users(mm.get_output(0))
    add.__class__ = trt.IElementWiseLayer
    assert add.op == trt.ElementWiseOperation.SUM
    b = constant(add.get_input(1-slot)).reshape(-1)
    assert b.shape == (plugin.N,) and b.dtype == np.float16
    w, b = np.ascontiguousarray(w), np.ascontiguousarray(b)
    keep += [w, b]
    custom = net.add_plugin(trtp.op.nano_vla.gemv_bias(
        x, net.add_constant(w.shape, trt.Weights(w)).get_output(0),
        net.add_constant(b.shape, trt.Weights(b)).get_output(0)), aot=True)
    custom.name = 'triton_gemv_' + mm.name
    new = custom.get_output(0)
    assert tuple(new.shape) == tuple(add.get_output(0).shape) and new.dtype == add.get_output(0).dtype
    for l, j in users(add.get_output(0)):
        l.set_input(j, new)
    patch = dict(matmul=mm.name, bias_add=add.name, op1=str(mm.op1))
config = builder.create_builder_config()
config.builder_optimization_level = 2
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
timing = base/'timing.cache'
config.set_timing_cache(config.create_timing_cache(timing.read_bytes() if timing.exists() else b''), ignore_mismatch=False)
started = time.time()
blob = builder.build_serialized_network(net, config)
assert blob is not None
(out/'action_output.engine').write_bytes(blob)
here = Path(__file__).parent
(out/'candidate.json').write_text(json.dumps(dict(
    mode=a.mode, bundle=str(bundle), base_cache=str(base), tensorrt=trt.__version__, patch=patch,
    tile=plugin.TILE, kernel_sha256=sha256(here/'kernels.py'), plugin_sha256=sha256(here/'plugin.py'),
    precision='FP32 accumulation and bias add, HALF in/out (TensorRT: FP16-accumulating tactic)',
    build_s=round(time.time()-started, 1), **pin(out, bundle)), indent=2))
print('BUILT', a.mode, patch)
