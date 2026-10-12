"""Rebuild EVO1's action_step engine with its cross-attention as a Triton AOT plugin.

Every step, the exported graph transposes the cached key [1,H,NK,D] -> [1,H,D,NK] for
QK^T and runs TensorRT's MHA on it; K and V only change per observation. The plugin
reads Q (from q-proj), K, V and the key mask in place. Each replaced chain is verified
first: the score is MatMul -> Mul(scalar) -> Add(mask broadcast over [NK]) -> Cast(FP32)
-> Softmax -> Cast(FP16) -> MatMul, and the index maps of Q, K, V and the output through
every reshape/transpose/slice match the strides the kernel uses. Other engines are
copied byte-identical from --base-cache (a plain or gemv candidate cache of the bundle).
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import tensorrt as trt
import tensorrt.plugin as trtp
import plugin
from candidate import pin, sha256
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bench.vendor.groot_trt import configure_accumulate

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['baseline', 'attention'], required=True)
p.add_argument('--bundle', type=Path, required=True)
p.add_argument('--base-cache', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--accumulate', choices=['fp32', 'auto'], default='fp32', help='as bench trt-split --accumulate')
p.add_argument('--graph-scores', action='store_true',
               help="reproduce the graph's FP16 score roundings instead of FP32 scores")
p.add_argument('--two-pass', action='store_true', help='round normalized probabilities, as the graph')
a = p.parse_args()
bundle, base, out = (x.expanduser().resolve() for x in (a.bundle, a.base_cache, a.out))
out.mkdir(parents=True, exist_ok=False)
for path in base.glob('*.engine'):
    if path.stem != 'action_step' and not path.name.startswith('op_'):
        shutil.copy2(path, out/path.name)
for path in base.glob('*.npz'):
    shutil.copy2(path, out/path.name)

T = trt.LayerType
CLASSES = {T.SHUFFLE: trt.IShuffleLayer, T.SLICE: trt.ISliceLayer, T.CONSTANT: trt.IConstantLayer,
           T.ELEMENTWISE: trt.IElementWiseLayer, T.CAST: trt.ICastLayer, T.SOFTMAX: trt.ISoftMaxLayer,
           T.MATRIX_MULTIPLY: trt.IMatrixMultiplyLayer}
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
parser = trt.OnnxParser(net, logger)
assert parser.parse_from_file(str(bundle/'action_step.onnx')), [str(parser.get_error(i)) for i in range(parser.num_errors)]
layers = [net.get_layer(i) for i in range(net.num_layers)]
producers = {l.get_output(j).name: l for l in layers for j in range(l.num_outputs)}


def cls(l):
    l.__class__ = CLASSES.get(l.type, l.__class__)
    return l


def users(t):
    return [(l, j) for l in layers for j in range(l.num_inputs)
            if l.get_input(j) is not None and l.get_input(j).name == t.name]


def constant(t):
    l = cls(producers[t.name])
    while l.type in (T.SHUFFLE, T.CAST):
        l = cls(producers[l.get_input(0).name])
    assert l.type == T.CONSTANT, l.name
    return np.asarray(l.weights).reshape(tuple(l.shape))


def emulate(l, x):
    """Apply a view layer to an index array (static shapes and constant parameters only)."""
    l = cls(l)
    if l.type == T.CAST:
        return x
    if l.type == T.SLICE:
        assert l.num_inputs == 1 and l.mode in (trt.SampleMode.STRICT_BOUNDS, trt.SampleMode.WRAP), l.name
        for axis, (s, n, k) in enumerate(zip(l.start, l.shape, l.stride)):
            ix = s + np.arange(n)*k
            if l.mode == trt.SampleMode.WRAP:
                ix = ix % x.shape[axis]
            x = np.take(x, ix, axis=axis)
        return x
    assert l.type == T.SHUFFLE, l.name
    x = x.transpose(tuple(l.first_transpose)[:x.ndim])
    if l.num_inputs > 1 and l.get_input(1) is not None:
        shape = tuple(int(v) for v in constant(l.get_input(1)).ravel())
    else:
        try:
            shape = tuple(l.reshape_dims)
        except ValueError:
            shape = ()
    if shape:
        if l.zero_is_placeholder:
            shape = tuple(x.shape[i] if d == 0 else d for i, d in enumerate(shape))
        x = x.reshape(shape)
    x = x.transpose(tuple(l.second_transpose)[:x.ndim])
    assert x.shape == tuple(l.get_output(0).shape), (l.name, x.shape)
    return x


def native(t):
    """(root tensor, index map of t into root's flat elements) through view layers."""
    chain = []
    while t.name in producers and producers[t.name].type in (T.SHUFFLE, T.SLICE, T.CAST):
        l = cls(producers[t.name])
        if l.type == T.CAST and t.dtype != l.get_input(0).dtype:
            break                       # a real conversion is not a view
        chain.append(l)
        t = l.get_input(0)
    idx = np.arange(int(np.prod(t.shape))).reshape(tuple(t.shape))
    for l in reversed(chain):
        idx = emulate(l, idx)
    return t, idx


def fit(m, name):
    """(head stride, row stride) of an index map [1,H,R,D] affine in the root; else refuse."""
    m = m[0]
    H, R, D = m.shape
    sh, ss = int(m[1, 0, 0] - m[0, 0, 0]), int(m[0, 1, 0] - m[0, 0, 0])
    expect = np.arange(H)[:, None, None]*sh + np.arange(R)[None, :, None]*ss + np.arange(D)[None, None, :]
    assert m[0, 0, 0] == 0 and np.array_equal(m, expect), (name, 'layout not affine')
    return sh, ss


patches, keep = [], []
if a.mode == 'attention':
    for soft in [cls(l) for l in layers if l.type == T.SOFTMAX]:
        cast = cls(producers[soft.get_input(0).name])
        assert cast.type == T.CAST and cast.get_input(0).dtype == trt.float16, soft.name
        add = cls(producers[cast.get_input(0).name])
        assert add.type == T.ELEMENTWISE and add.op == trt.ElementWiseOperation.SUM, soft.name
        x, mask = add.get_input(0), add.get_input(1)
        if cls(producers[x.name]).type != T.ELEMENTWISE:
            x, mask = mask, x
        mul = cls(producers[x.name])
        assert mul.type == T.ELEMENTWISE and mul.op == trt.ElementWiseOperation.PROD
        left, c = mul.get_input(0), mul.get_input(1)
        if np.prod(c.shape) != 1:
            left, c = c, left
        scale = float(constant(c).ravel()[0])
        mm = cls(producers[left.name])
        assert mm.type == T.MATRIX_MULTIPLY and mm.op0 == mm.op1 == trt.MatrixOperation.NONE
        _, H, NQ, D = tuple(mm.get_input(0).shape)
        NK = tuple(mm.get_input(1).shape)[-1]
        q, qmap = native(mm.get_input(0))
        k, kmap = native(mm.get_input(1))
        mroot, mmap = native(mask)
        assert mroot.dtype == q.dtype == k.dtype == trt.float16
        # The mask must be one row of NK values broadcast over heads and queries.
        mmap = np.broadcast_to(mmap, (1, H, NQ, NK)) if mmap.shape != (1, H, NQ, NK) else mmap
        assert int(np.prod(mroot.shape)) == NK and np.array_equal(mmap, np.broadcast_to(np.arange(NK), mmap.shape)), 'mask broadcast'
        prob, dtypes = soft.get_output(0), []
        while True:                 # FP32 softmax -> FP16 cast, identity reshapes allowed
            (l, slot), = users(prob)
            l = cls(l)
            if l.type == T.MATRIX_MULTIPLY:
                pv = l
                break
            if l.type == T.SHUFFLE:
                ids = np.arange(int(np.prod(prob.shape))).reshape(tuple(prob.shape))
                assert np.array_equal(emulate(l, ids), ids), (l.name, 'non-identity shuffle')
            else:
                assert l.type == T.CAST, (l.name, l.type)
                dtypes.append(l.get_output(0).dtype)
            prob = l.get_output(0)
        assert dtypes and dtypes[0] == dtypes[-1] == trt.float16 and set(dtypes) <= {trt.float16, trt.float32}
        assert slot == 0 and pv.op0 == pv.op1 == trt.MatrixOperation.NONE
        v, vmap = native(pv.get_input(1))
        sqh, sqs = fit(qmap, 'Q')
        skh, sks = fit(kmap.transpose(0, 1, 3, 2), 'K')
        svh, svs = fit(vmap, 'V')
        output = pv.get_output(0)
        outmap = np.arange(H*NQ*D).reshape(1, H, NQ, D)
        while tuple(output.shape) != (1, NQ, H*D):
            (l, _), = users(output)
            outmap = emulate(l, outmap)
            output = l.get_output(0)
        assert np.array_equal(outmap, np.arange(H*NQ*D).reshape(1, H, NQ, D).transpose(0, 2, 1, 3).reshape(1, NQ, H*D)), 'output layout'
        key = plugin.register_step(NQ, NK, H, D, scale, (sqh, sqs, skh, sks, svh, svs),
                                   score_fp32=not a.graph_scores, two_pass=a.two_pass)
        custom = net.add_plugin(getattr(trtp.op.nano_vla, key)(q, k, v, mroot), aot=True)
        custom.name = 'triton_' + soft.name
        for l, j in users(output):
            l.set_input(j, custom.get_output(0))
        patches.append(dict(softmax=soft.name, shape=key, q=q.name, k=k.name, v=v.name, mask=mroot.name))
    assert patches, 'no cross-attention matched'

config = builder.create_builder_config()
config.builder_optimization_level = 2
accumulated = configure_accumulate(net, config, a.accumulate)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
timing = base/'timing.cache'
config.set_timing_cache(config.create_timing_cache(timing.read_bytes() if timing.exists() else b''), ignore_mismatch=False)
started = time.time()
blob = builder.build_serialized_network(net, config)
assert blob is not None
(out/'action_step.engine').write_bytes(blob)
here = Path(__file__).parent
base_manifest = json.loads((base/'candidate.json').read_text()) if (base/'candidate.json').exists() else None
(out/'candidate.json').write_text(json.dumps(dict(
    mode=a.mode, accumulate=a.accumulate, score_fp32=not a.graph_scores, two_pass=a.two_pass,
    fp32_accumulated_matmuls=accumulated, bundle=str(bundle), base_cache=str(base), base_manifest=base_manifest,
    tensorrt=trt.__version__, patches=patches, tile=plugin.STEP_TILE,
    kernel_sha256=sha256(here/'kernels.py'), plugin_sha256=sha256(here/'plugin.py'),
    build_s=round(time.time()-started, 1), **pin(out, bundle)), indent=2))
print('BUILT', a.mode, len(patches), 'patches')
