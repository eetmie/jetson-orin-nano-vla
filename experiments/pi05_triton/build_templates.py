"""Rebuild π0.5's action template with the Triton attention plugin into a new cache.

The template stays weight-stripped and individually refittable (the plugin has no
weights); every other template is copied byte-identical from --base-cache. Each link
of the exported attention chain is checked before rewiring.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import tensorrt as trt
import tensorrt.plugin as trtp
import plugin

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['baseline', 'attention'], required=True)
p.add_argument('--bundle', type=Path, default=Path.home()/'bundles/pi05-libero-compact-split')
p.add_argument('--base-cache', type=Path, default=Path.home()/'.cache/jetson-orin-nano-vla/pi05-libero-compact-trt')
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
bundle, base, out = (x.expanduser().resolve() for x in (a.bundle, a.base_cache, a.out))
meta = json.loads((bundle/'bundle.json').read_text())
out.mkdir(parents=True, exist_ok=False)
for path in base.iterdir():
    if path.suffix in ('.engine', '.sha256') and not path.name.startswith('action.'):
        shutil.copy2(path, out/path.name)
T = trt.LayerType
CLS = {T.SHUFFLE: trt.IShuffleLayer, T.SLICE: trt.ISliceLayer, T.CAST: trt.ICastLayer,
       T.CONSTANT: trt.IConstantLayer, T.ELEMENTWISE: trt.IElementWiseLayer,
       T.MATRIX_MULTIPLY: trt.IMatrixMultiplyLayer, T.CONCATENATION: trt.IConcatenationLayer}
onnx_path = bundle/'templates'/'action.onnx'
descriptor = json.loads(onnx_path.with_suffix('.json').read_text())
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
parser = trt.OnnxParser(net, logger)
assert parser.parse_from_file(str(onnx_path)), [str(parser.get_error(i)) for i in range(parser.num_errors)]
layers = [net.get_layer(i) for i in range(net.num_layers)]
producers = {l.get_output(j).name: l for l in layers for j in range(l.num_outputs)}


def cls(l):
    l.__class__ = CLS.get(l.type, l.__class__)
    return l


def users(t):
    return [(l, j) for l in layers for j in range(l.num_inputs) if l.get_input(j) is not None and l.get_input(j).name == t.name]


def single_user(t, kind):
    (l, j), = users(t)
    l = cls(l)
    assert l.type == kind, (l.name, l.type, kind)
    return l, j


def through_cast(t, to):
    l = cls(producers[t.name])
    assert l.type == T.CAST and l.get_output(0).dtype == to, (l.name, l.type)
    return l.get_input(0)


def emulate(layer, arr):
    """Index map of a parsed shuffle / slice / (broadcast) expand layer."""
    layer = cls(layer)
    outshape = tuple(layer.get_output(0).shape)
    if layer.type == T.SLICE and any(layer.get_input(i) is not None for i in range(1, layer.num_inputs)):
        # The ONNX parser's Expand: a slice with runtime shape inputs and stride 0 on the
        # broadcast dims. Its output shape is static here, so it is a plain broadcast.
        assert 'Expand' in layer.name and arr.ndim == len(outshape), layer.name
        assert all(a in (1, o) for a, o in zip(arr.shape, outshape)), (layer.name, arr.shape, outshape)
        return np.broadcast_to(arr, outshape)
    if layer.type == T.SLICE:
        idx = [np.asarray(s) + np.arange(n)*k for s, n, k in zip(layer.start, layer.shape, layer.stride)]
        return arr[np.ix_(*idx)]
    if layer.type in (T.SQUEEZE, T.UNSQUEEZE):
        return arr.reshape(outshape)
    assert layer.type == T.SHUFFLE and (layer.num_inputs == 1 or layer.get_input(1) is None), layer.name
    arr = arr.transpose(tuple(layer.first_transpose)[:arr.ndim])
    try:
        dims = tuple(layer.reshape_dims)
    except ValueError:
        dims = ()
    if dims:
        arr = arr.reshape(tuple(arr.shape[i] if d == 0 and layer.zero_is_placeholder else d for i, d in enumerate(dims)))
    arr = arr.transpose(tuple(layer.second_transpose)[:arr.ndim])
    assert tuple(arr.shape) == outshape, (layer.name, arr.shape, outshape)
    return arr


def back_to(t, shape):
    """Walk index-only layers back to a tensor of `shape`; return it and the index map."""
    chain = []
    while tuple(t.shape) != shape:
        l = cls(producers[t.name])
        assert l.type in (T.SHUFFLE, T.SLICE, T.SQUEEZE, T.UNSQUEEZE), (l.name, l.type)
        chain.append(l)
        t = l.get_input(0)
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    arr = idx
    for l in reversed(chain):
        arr = emulate(l, arr)
    return t, idx, arr


H, D, NQ, NKV = plugin.HEADS, plugin.DIM, plugin.N_Q, plugin.N_KV
patch = None
if a.mode == 'attention':
    (soft,) = [l for l in layers if l.type == T.SOFTMAX]
    add = cls(producers[soft.get_input(0).name])
    assert add.type == T.ELEMENTWISE and add.op == trt.ElementWiseOperation.SUM
    logits, mask = add.get_input(0), add.get_input(1)
    if cls(producers[logits.name]).type != T.CAST:
        logits, mask = mask, logits
    assert mask.dtype == trt.float32 and tuple(mask.shape) == (1, 1, NQ, NKV)
    scaled = through_cast(logits, trt.float32)
    mul = cls(producers[scaled.name])
    assert mul.type == T.ELEMENTWISE and mul.op == trt.ElementWiseOperation.PROD
    def constant(t):
        l = cls(producers[t.name])
        while l.type in (T.CAST, T.SHUFFLE):
            l = cls(producers[l.get_input(0).name])
        return l if l.type == T.CONSTANT else None
    s16, c = mul.get_input(0), mul.get_input(1)
    if constant(c) is None:
        s16, c = c, s16
    assert c.dtype == trt.float16 and float(np.asarray(constant(c).weights).reshape(-1)[0]) == plugin.SCALE
    qk = cls(producers[through_cast(s16, trt.float16).name])
    assert qk.type == T.MATRIX_MULTIPLY and qk.op0 == qk.op1 == trt.MatrixOperation.NONE
    q = through_cast(qk.get_input(0), trt.float32)
    assert tuple(q.shape) == (1, H, NQ, D) and q.dtype == trt.float16
    k, k_idx, k_map = back_to(through_cast(qk.get_input(1), trt.float32), (1, 1, NKV, D))
    assert np.array_equal(k_map, np.broadcast_to(k_idx.transpose(0, 1, 3, 2), (1, H, D, NKV))), 'K is not expand + transpose'
    probs = soft.get_output(0)
    seen_half = False
    while True:                 # parser shape shims, FP32 identity cast, HALF rounding, FP32 for PV
        (l, _), = users(probs)
        l = cls(l)
        assert l.type in (T.CAST, T.SHUFFLE), (l.name, l.type)
        if l.type == T.SHUFFLE:
            assert tuple(l.get_output(0).shape) == tuple(probs.shape), l.name
        elif l.get_output(0).dtype == trt.float16:
            seen_half = True
        probs = l.get_output(0)
        nxt = users(probs)
        if len(nxt) == 1 and cls(nxt[0][0]).type == T.MATRIX_MULTIPLY:
            break
    pv, slot = nxt[0]
    assert seen_half and probs.dtype == trt.float32, 'probabilities must round to HALF before PV'
    assert slot == 0 and pv.op0 == pv.op1 == trt.MatrixOperation.NONE
    v, v_idx, v_map = back_to(through_cast(pv.get_input(1), trt.float32), (1, 1, NKV, D))
    assert np.array_equal(v_map, np.broadcast_to(v_idx, (1, H, NKV, D))), 'V is not an expand'
    o16, _ = single_user(pv.get_output(0), T.CAST)
    t, tail = o16.get_output(0), []
    while tuple(t.shape) != (1, NQ, H*D):
        l, _ = single_user(t, T.SHUFFLE)
        tail.append(l)
        t = l.get_output(0)
    idx = np.arange(H*NQ*D).reshape(1, H, NQ, D)
    arr = idx
    for l in tail:
        arr = emulate(l, arr)
    assert np.array_equal(arr, idx.transpose(0, 2, 1, 3).reshape(1, NQ, H*D))
    custom = net.add_plugin(trtp.op.nano_vla.pi05_action_attention(q, k, v, mask), aot=True)
    custom.name = 'triton_pi05_action_attention'
    flat = net.add_shuffle(custom.get_output(0))
    flat.reshape_dims = (1, NQ, H*D)
    new = flat.get_output(0)
    for l, j in users(t):
        l.set_input(j, new)
    patch = dict(softmax=soft.name, q=q.name, k=k.name, v=v.name, mask=mask.name, replaced=t.name)
cfg = builder.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
cfg.builder_optimization_level = 2
cfg.clear_flag(trt.BuilderFlag.TF32)
cfg.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)
cfg.set_flag(trt.BuilderFlag.STRIP_PLAN)
for wt in descriptor['weights']:
    assert net.mark_weights_refittable(wt['name']), wt['name']
started = time.time()
plan = builder.build_serialized_network(net, cfg)
assert plan is not None
(out/'action.engine').write_bytes(plan)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
here = Path(__file__).parent
(out/'candidate.json').write_text(json.dumps(dict(
    mode=a.mode, bundle=str(bundle), base_cache=str(base), tensorrt=trt.__version__, patch=patch, tile=plugin.TILE,
    template_sha256={p.name: sha(p) for p in sorted((bundle/'templates').glob('*.onnx'))},
    engine_sha256={p.name: sha(p) for p in sorted(out.glob('*.engine')) if not p.name.startswith('op_')},
    kernel_sha256=sha(here/'kernels.py'), plugin_sha256=sha(here/'plugin.py'),
    precision='HALF QK (FP32 accumulation), exact 1/16 scale, FP32 mask+softmax, normalized probabilities '
              'rounded to HALF, HALF PV (FP32 accumulation): the exported order, two passes over keys',
    build_s=round(time.time()-started, 1)), indent=2))
print('BUILT', a.mode, patch)
