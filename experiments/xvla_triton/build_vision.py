"""Rebuild X-VLA's vision engines with DaViT's depthwise convs in token layout.

Every 3x3 depthwise conv + residual whose input comes from tokens through a transpose
and whose output goes back to tokens through one becomes `dwconv_tokens` on the token
tensors (no NCHW round trip). Convs fed by an engine input or a strided conv stay.
Other engines (including the patched denoiser) are copied from --base-cache, a verified
candidate cache of the same bundle.
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
from candidate import pin, sha256, verify
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.vendor.groot_trt import configure_accumulate

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['baseline', 'dwconv'], required=True)
p.add_argument('--bundle', type=Path, required=True)
p.add_argument('--base-cache', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--accumulate', choices=['fp32', 'auto'], default='fp32')
a = p.parse_args()
bundle, base, out = (x.expanduser().resolve() for x in (a.bundle, a.base_cache, a.out))
base_manifest = verify(base, bundle)
meta = json.loads((bundle/'bundle.json').read_text())
vision = [g['name'] for g in meta['graphs'] if g['name'].startswith('vision')]
out.mkdir(parents=True, exist_ok=False)
for path in base.glob('*.engine'):
    if path.stem not in vision and not path.name.startswith('op_'):
        shutil.copy2(path, out/path.name)
T = trt.LayerType
CLS = {T.SHUFFLE: trt.IShuffleLayer, T.CONVOLUTION: trt.IConvolutionLayer, T.ELEMENTWISE: trt.IElementWiseLayer}


def build(name):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, logger)
    assert parser.parse_from_file(str(bundle/f'{name}.onnx'))
    layers = [net.get_layer(i) for i in range(net.num_layers)]
    producers = {l.get_output(j).name: l for l in layers for j in range(l.num_outputs)}
    inputs = {net.get_input(i).name for i in range(net.num_inputs)}

    def cls(l):
        l.__class__ = CLS.get(l.type, l.__class__)
        return l

    def users(t):
        return [(l, j) for l in layers for j in range(l.num_inputs) if l.get_input(j) is not None and l.get_input(j).name == t.name]

    def emulate(layer, arr):
        layer = cls(layer)
        outshape = tuple(layer.get_output(0).shape)
        assert layer.type == T.SHUFFLE, layer.name
        arr = arr.transpose(tuple(layer.first_transpose)[:arr.ndim])
        if layer.num_inputs > 1 and layer.get_input(1) is not None:
            arr = arr.reshape(outshape) if arr.ndim != len(outshape) or arr.shape != outshape else arr
        else:
            try:
                dims = tuple(layer.reshape_dims)
            except ValueError:
                dims = ()
            if dims:
                arr = arr.reshape(tuple(arr.shape[i] if d == 0 and layer.zero_is_placeholder else d for i, d in enumerate(dims)))
        arr = arr.transpose(tuple(layer.second_transpose)[:arr.ndim])
        assert tuple(arr.shape) == outshape, (layer.name, arr.shape, outshape)
        return arr

    patches, keep = [], []
    for conv in [cls(l) for l in layers if l.type == T.CONVOLUTION] if a.mode == 'dwconv' else []:
        x = conv.get_input(0)
        v, c, h, w = tuple(x.shape)
        if not (conv.num_groups == c and tuple(conv.kernel_size_nd) == (3, 3) and tuple(conv.stride_nd) == (1, 1)
                and tuple(conv.padding_nd) == (1, 1) and tuple(conv.dilation_nd) == (1, 1)):
            continue
        # input side: tokens [V, HW, C] -> shuffles -> x [V, C, H, W]
        chain, t = [], x
        while t.name not in inputs and t.name in producers and cls(producers[t.name]).type == T.SHUFFLE:
            chain.append(producers[t.name])
            t = producers[t.name].get_input(0)
            if tuple(t.shape) == (v, h*w, c):
                break
        if tuple(t.shape) != (v, h*w, c):
            continue
        tok = np.arange(v*h*w*c).reshape(v, h*w, c)
        arr = tok
        for l in reversed(chain):
            arr = emulate(l, arr)
        if not np.array_equal(arr, tok.transpose(0, 2, 1).reshape(v, c, h, w)):
            continue
        # residual: x + conv(x)
        (add, slot), = users(conv.get_output(0))
        add = cls(add)
        assert add.type == T.ELEMENTWISE and add.op == trt.ElementWiseOperation.SUM, add.name
        assert add.get_input(1-slot).name == x.name, add.name
        # output side: add -> shuffles -> tokens [V, HW, C] (Shape users only read dims)
        tail, u = [], add.get_output(0)
        while tuple(u.shape) != (v, h*w, c) or not tail:
            nxt = [(l, j) for l, j in users(u) if l.type != T.SHAPE]
            if len(nxt) != 1 or cls(nxt[0][0]).type != T.SHUFFLE or nxt[0][1] != 0:
                tail = None
                break
            tail.append(nxt[0][0])
            u = nxt[0][0].get_output(0)
        if tail is None:
            continue
        nchw = np.arange(v*c*h*w).reshape(v, c, h, w)
        arr = nchw
        for l in tail:
            arr = emulate(l, arr)
        if not np.array_equal(arr, nchw.reshape(v, c, h*w).transpose(0, 2, 1)):
            continue
        k = np.asarray(conv.kernel).reshape(c, 9).astype(np.float16)
        bias = np.asarray(conv.bias).reshape(c).astype(np.float16)
        assert np.asarray(conv.kernel).dtype == np.float16 and np.asarray(conv.bias).dtype == np.float16
        k, bias = np.ascontiguousarray(k), np.ascontiguousarray(bias)
        keep += [k, bias]
        assert (v, h*w, w, c) in plugin.DWCONV_SHAPES, (v, h, w, c)
        op = getattr(trtp.op.nano_vla, plugin.dwconv_op(v, h*w, w, c))
        custom = net.add_plugin(op(t, net.add_constant(k.shape, trt.Weights(k)).get_output(0),
                                   net.add_constant(bias.shape, trt.Weights(bias)).get_output(0)), aot=True)
        custom.name = 'triton_dwconv_' + conv.name
        for l, j in users(u):
            l.set_input(j, custom.get_output(0))
        patches.append(dict(conv=conv.name, tokens_in=t.name, tokens_out=u.name, shape=[v, h, w, c]))
    config = builder.create_builder_config()
    config.builder_optimization_level = 2
    accumulated = configure_accumulate(net, config, a.accumulate)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    timing = base/'timing.cache'
    config.set_timing_cache(config.create_timing_cache(timing.read_bytes() if timing.exists() else b''), ignore_mismatch=False)
    started = time.time()
    blob = builder.build_serialized_network(net, config)
    assert blob is not None, name
    (out/f'{name}.engine').write_bytes(blob)
    print('BUILT', name, len(patches), 'dwconv patches', accumulated, 'fp32-accumulated', round(time.time()-started, 1), 's', flush=True)
    return patches


patches = {n: build(n) for n in vision}
here = Path(__file__).parent
manifest = dict(base_manifest)
manifest.update(vision_mode=a.mode, vision_accumulate=a.accumulate, vision_patches=patches,
                vision_kernel_sha256=sha256(here/'kernels.py'), vision_plugin_sha256=sha256(here/'plugin.py'),
                base_cache=str(base), **pin(out, bundle))
(out/'candidate.json').write_text(json.dumps(manifest, indent=2))
print('DONE', sum(len(v) for v in patches.values()), 'patches')
