"""Rebuild X-VLA's denoise engines with AOT Triton attention into a new, isolated cache.

baseline   the unchanged graphs with the same builder settings (control)
attention  every block's QK/softmax/PV chain -> one plugin on the fused QKV projection
actions    attention, plus the last block computes only the action rows that are decoded

Every link of the exported chain is checked before rewiring. Other engines are copied
byte-identical from --base-cache, a plain prebuilt cache of the same bundle.
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
from candidate import pin, sha256

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['baseline', 'attention', 'actions'], required=True)
p.add_argument('--bundle', type=Path, default=Path.home()/'bundles/xvla-base-split')
p.add_argument('--base-cache', type=Path, default=Path.home()/'.cache/jetson-orin-nano-vla/xvla-base-trt')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--opt-level', type=int, default=2)
a = p.parse_args()
bundle, base, out = (x.expanduser().resolve() for x in (a.bundle, a.base_cache, a.out))
meta = json.loads((bundle/'bundle.json').read_text())
denoise = [g['name'] for g in meta['graphs'] if g['name'].startswith('denoise')]
out.mkdir(parents=True, exist_ok=False)
for path in base.glob('*.engine'):
    if path.stem not in denoise and not path.name.startswith('op_'):
        shutil.copy2(path, out/path.name)

T = trt.LayerType
NP = {trt.float16: np.float16, trt.float32: np.float32, trt.int64: np.int64, trt.int32: np.int32}
SEQ, WIDTH = plugin.SEQ, plugin.HEADS*plugin.DIM


def cls(layer):
    layer.__class__ = {T.SHUFFLE: trt.IShuffleLayer, T.SLICE: trt.ISliceLayer, T.CONSTANT: trt.IConstantLayer,
                       T.ELEMENTWISE: trt.IElementWiseLayer, T.UNARY: trt.IUnaryLayer, T.CAST: trt.ICastLayer,
                       T.MATRIX_MULTIPLY: trt.IMatrixMultiplyLayer}.get(layer.type, layer.__class__)
    return layer


def build(name, actions):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, logger)
    assert parser.parse_from_file(str(bundle/f'{name}.onnx')), [str(parser.get_error(i)) for i in range(parser.num_errors)]
    layers = [net.get_layer(i) for i in range(net.num_layers)]
    producers = {l.get_output(j).name: l for l in layers for j in range(l.num_outputs)}

    def users(t):
        return [(l, j) for l in layers for j in range(l.num_inputs)
                if l.get_input(j) is not None and l.get_input(j).name == t.name]

    def evaluate(t):
        """Numpy value of a constant-only subgraph (the exported attention scale)."""
        l = cls(producers[t.name])
        ins = [l.get_input(i) for i in range(l.num_inputs)]
        if l.type == T.CONSTANT:
            return np.asarray(l.weights).reshape(tuple(l.shape))
        if l.type == T.SHAPE:
            return np.array(tuple(ins[0].shape), np.int64)
        x = evaluate(ins[0])
        if l.type == T.SLICE:
            assert all(i is None for i in ins[1:])
            return x[tuple(slice(s, s+n*k, k) for s, n, k in zip(l.start, l.shape, l.stride))]
        if l.type == T.CAST:
            return x.astype(NP[l.get_output(0).dtype])
        if l.type == T.SHUFFLE:
            return x.reshape(tuple(l.get_output(0).shape))
        if l.type == T.UNARY:
            assert l.op == trt.UnaryOperation.SQRT, l.op
            return np.sqrt(x).astype(x.dtype)
        if l.type == T.ELEMENTWISE:
            y = evaluate(ins[1])
            op = {trt.ElementWiseOperation.DIV: np.divide, trt.ElementWiseOperation.PROD: np.multiply,
                  trt.ElementWiseOperation.SUM: np.add, trt.ElementWiseOperation.SUB: np.subtract}[l.op]
            return op(x, y).astype(NP[l.get_output(0).dtype])
        raise AssertionError(f'{l.name}: {l.type} in a constant subgraph')

    def sdpa_scale(t, head_dim):
        """The exported SDPA scale: Sqrt(Cast(Div(1, Sqrt(Cast(Slice(Shape(q))[-1:]))))) in HALF.

        The Slice reads q's last dim through shape tensors TensorRT cannot evaluate here,
        so check the structure (the ONNX slices [-1:]) and compute it from head_dim."""
        def through(t, *kinds):
            l = cls(producers[t.name])
            while l.type == T.SHUFFLE:
                l = cls(producers[l.get_input(0).name])
            assert l.type in kinds, (l.name, l.type)
            return l
        sq = through(t, T.UNARY)
        assert sq.op == trt.UnaryOperation.SQRT
        div = through(through(sq.get_input(0), T.CAST).get_input(0), T.ELEMENTWISE)
        assert div.op == trt.ElementWiseOperation.DIV
        one = evaluate(div.get_input(0))
        assert one.size == 1 and float(one.reshape(-1)[0]) == 1.0
        inner = through(div.get_input(1), T.UNARY)
        assert inner.op == trt.UnaryOperation.SQRT
        through(through(inner.get_input(0), T.CAST).get_input(0), T.SLICE)
        shared_div.add(div.name)
        h = np.float16
        return float(h(np.sqrt(h(h(1.0)/h(np.sqrt(h(head_dim)))))))

    def emulate(layer, arr):
        layer = cls(layer)
        if layer.type in (T.SQUEEZE, T.UNSQUEEZE):          # no data movement
            return arr.reshape(tuple(layer.get_output(0).shape))
        if layer.type == T.SLICE:
            assert all(layer.get_input(i) is None for i in range(1, layer.num_inputs)), layer.name
            return arr[tuple(slice(s, s+n*k, k) for s, n, k in zip(layer.start, layer.shape, layer.stride))]
        assert layer.type == T.SHUFFLE and (layer.num_inputs == 1 or layer.get_input(1) is None), layer.name
        arr = arr.transpose(tuple(layer.first_transpose)[:arr.ndim])
        try:
            dims = tuple(layer.reshape_dims)
        except ValueError:
            dims = ()
        if dims:
            arr = arr.reshape(tuple(arr.shape[i] if d == 0 and layer.zero_is_placeholder else d for i, d in enumerate(dims)))
        arr = arr.transpose(tuple(layer.second_transpose)[:arr.ndim])
        assert tuple(arr.shape) == tuple(layer.get_output(0).shape), (layer.name, arr.shape)
        return arr

    def back_to_qkv(t):
        """Walk a Q/K^T/V operand back to the [1,S,3*W] projection; return (qkv, index map, scale)."""
        chain, scale = [], None
        while tuple(t.shape) != (1, SEQ, 3*WIDTH):
            l = cls(producers[t.name])
            if l.type in (T.SHUFFLE, T.SLICE, T.SQUEEZE, T.UNSQUEEZE):
                chain.append(l)
                t = l.get_input(0)
                continue
            assert l.type == T.ELEMENTWISE and l.op == trt.ElementWiseOperation.PROD and scale is None, l.name
            x, c = l.get_input(0), l.get_input(1)
            if tuple(c.shape) != (1, 1, 1, 1):
                x, c = c, x
            scale, t = sdpa_scale(c, plugin.DIM), x
        idx = np.arange(SEQ*3*WIDTH).reshape(1, SEQ, 3*WIDTH)
        arr = idx
        for l in reversed(chain):
            arr = emulate(l, arr)
        return t, arr, scale

    softmaxes = [l for l in layers if l.type == T.SOFTMAX]
    assert len(softmaxes) == 6, len(softmaxes)
    split = np.arange(SEQ*3*WIDTH).reshape(1, SEQ, 3, plugin.HEADS, plugin.DIM)
    patches, last_residual = [], None
    for i, soft in enumerate(softmaxes if a.mode != 'baseline' else []):
        shared_div = set()      # Q's and K's scale must come from one Div on Q's head dim
        cast_in = cls(producers[soft.get_input(0).name])
        assert cast_in.type == T.CAST and cast_in.get_input(0).dtype == trt.float16
        qk = cls(producers[cast_in.get_input(0).name])
        assert qk.type == T.MATRIX_MULTIPLY and qk.op0 == qk.op1 == trt.MatrixOperation.NONE
        probs = soft.get_output(0)
        while True:                                  # softmax shim shuffles, then the HALF cast
            (l, _), = users(probs)
            l = cls(l)
            probs = l.get_output(0)
            if l.type == T.CAST:
                assert probs.dtype == trt.float16
                break
            assert l.type == T.SHUFFLE and tuple(probs.shape) == tuple(soft.get_output(0).shape)
        (pv, slot), = users(probs)
        pv = cls(pv)
        assert pv.type == T.MATRIX_MULTIPLY and slot == 0 and pv.op0 == pv.op1 == trt.MatrixOperation.NONE
        qkv_q, map_q, s_q = back_to_qkv(qk.get_input(0))
        qkv_k, map_k, s_k = back_to_qkv(qk.get_input(1))
        qkv_v, map_v, s_v = back_to_qkv(pv.get_input(1))
        assert qkv_q.name == qkv_k.name == qkv_v.name
        assert len(shared_div) == 1 and tuple(qk.get_input(0).shape)[-1] == plugin.DIM, shared_div
        assert np.array_equal(map_q, split[:, :, 0].transpose(0, 2, 1, 3))
        assert np.array_equal(map_k, split[:, :, 1].transpose(0, 2, 3, 1))
        assert np.array_equal(map_v, split[:, :, 2].transpose(0, 2, 1, 3))
        assert s_q == s_k == plugin.SCALE and s_v is None, (s_q, s_k, s_v)
        t, tail = pv.get_output(0), []
        while tuple(t.shape) != (1, SEQ, WIDTH):     # transpose + reshape back to [1,S,W]
            (l, _), = users(t)
            tail.append(l)
            t = l.get_output(0)
        idx = np.arange(SEQ*WIDTH).reshape(1, plugin.HEADS, SEQ, plugin.DIM)
        arr = idx
        for l in tail:
            arr = emulate(l, arr)
        assert np.array_equal(arr, idx.transpose(0, 2, 1, 3).reshape(1, SEQ, WIDTH))
        downstream = users(t)
        if actions and i == len(softmaxes)-1:
            # Last block: only rows [:ACTIONS] reach the decoder (the graph slices after
            # it), and every op after attention is row-wise. Keys/values keep all rows.
            (proj, _), = downstream
            proj = cls(proj)
            assert proj.type == T.MATRIX_MULTIPLY
            (bias, _), = users(proj.get_output(0))
            bias = cls(bias)
            assert bias.type == T.ELEMENTWISE and bias.op == trt.ElementWiseOperation.SUM
            (resid, slot_r), = users(bias.get_output(0))
            resid = cls(resid)
            assert resid.type == T.ELEMENTWISE and resid.op == trt.ElementWiseOperation.SUM
            x = resid.get_input(1-slot_r)
            assert tuple(x.shape) == (1, SEQ, WIDTH)
            rows = net.add_slice(x, (0, 0, 0), (1, plugin.ACTIONS, WIDTH), (1, 1, 1)).get_output(0)
            custom = net.add_plugin(trtp.op.nano_vla.xvla_attention_actions(qkv_q, rows), aot=True)
            resid.set_input(1-slot_r, rows)
            last_residual = resid.name
        else:
            custom = net.add_plugin(trtp.op.nano_vla.xvla_attention(qkv_q), aot=True)
        custom.name = f'triton_xvla_attention_{soft.name}'
        new = custom.get_output(0)
        for l, j in downstream:
            l.set_input(j, new)
        patches.append(dict(softmax=soft.name, qkv=qkv_q.name, replaced=t.name, rows=tuple(new.shape)[1]))
    if actions:
        assert last_residual is not None
        outputs = [net.get_output(i) for i in range(net.num_outputs)]
        assert [tuple(o.shape) for o in outputs] == [(1, plugin.ACTIONS, meta['max_action_dim'])], [o.shape for o in outputs]
    config = builder.create_builder_config()
    config.builder_optimization_level = a.opt_level
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    timing = base/'timing.cache'
    config.set_timing_cache(config.create_timing_cache(timing.read_bytes() if timing.exists() else b''), ignore_mismatch=False)
    started = time.time()
    blob = builder.build_serialized_network(net, config)
    assert blob is not None, f'{name}: build failed'
    (out/f'{name}.engine').write_bytes(blob)
    print('BUILT', name, len(patches), 'patches', round(time.time()-started, 1), 's', flush=True)
    return patches


patches = {n: build(n, a.mode == 'actions' and n == denoise[-1]) for n in denoise}
here = Path(__file__).parent
manifest = dict(mode=a.mode, bundle=str(bundle), base_cache=str(base), tensorrt=trt.__version__,
                opt_level=a.opt_level, tile=plugin.TILE, scale=plugin.SCALE, patches=patches,
                kernel_sha256=sha256(here/'kernels.py'), plugin_sha256=sha256(here/'plugin.py'),
                precision='HALF Q/K scaling and score rounding, FP32 online softmax and accumulation, '
                          'unnormalized probabilities rounded to HALF before PV', **pin(out, bundle))
(out/'candidate.json').write_text(json.dumps(manifest, indent=2))
print('DONE', a.mode, out)
