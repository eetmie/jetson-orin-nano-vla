"""Build an isolated expert candidate on top of a verified vision candidate.

Baseline rebuilds the unchanged expert; ffn replaces all 16 gated projections.
Other engines, including the improved vision engine, retain identical bytes.
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
import ffn_plugin
from candidate_cache import sha256, verify_candidate
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.vendor.groot_trt import configure_accumulate

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['baseline','ffn'], required=True)
parser.add_argument('--base-cache', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--bundle', type=Path, default=Path.home()/'bundles/smolvla-base-split')
parser.add_argument('--opt-level', type=int, default=2, help='TensorRT builder optimization level')
parser.add_argument('--accumulate', choices=['fp32', 'auto'], default='fp32', help='as bench trt-split --accumulate')
args = parser.parse_args()
base = args.base_cache.expanduser().resolve()
bundle = args.bundle.expanduser().resolve()
base_manifest = verify_candidate(base, bundle)
dest = args.out.expanduser().resolve()
dest.mkdir(parents=True, exist_ok=False)
for path in base.glob('*.engine'):
    if path.name != 'decode.engine':
        shutil.copy2(path, dest/path.name)
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
net = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
source = bundle/'smolvlm_expert_decode.onnx'
onnx_parser = trt.OnnxParser(net, logger)
assert onnx_parser.parse_from_file(str(source)), [str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)]
layers = [net.get_layer(i) for i in range(net.num_layers)]
producers = {l.get_output(i).name:l for l in layers for i in range(l.num_outputs)}
def users(t):
    return [(l,i) for l in layers for i in range(l.num_inputs)
            if l.get_input(i) is not None and l.get_input(i).name == t.name]
def product(l):
    assert l.type == trt.LayerType.ELEMENTWISE, l.name
    l.__class__ = trt.IElementWiseLayer
    assert l.op == trt.ElementWiseOperation.PROD, l.name
def constant(t):
    expected = tuple(t.shape)
    layer = producers[t.name]
    if layer.type == trt.LayerType.SHUFFLE:
        layer.__class__ = trt.IShuffleLayer
        assert list(layer.first_transpose)[:3] == [0,1,2]
        assert list(layer.second_transpose)[:3] == [0,1,2]
        layer = producers[layer.get_input(0).name]
    assert layer.type == trt.LayerType.CONSTANT, layer.name
    layer.__class__ = trt.IConstantLayer
    values = np.asarray(layer.weights)
    assert values.dtype == np.float16 and values.size == 720*2048
    assert expected == (1,720,2048)
    return values.reshape(720,2048)

patches, weight_buffers = [], []
sigmoids = []
for l in layers:
    if l.type == trt.LayerType.ACTIVATION:
        l.__class__ = trt.IActivationLayer
        if l.type == trt.ActivationType.SIGMOID:
            sigmoids.append(l)
assert len(sigmoids) == 16
if args.mode == 'ffn':
    for sigmoid in sigmoids:
        gate = sigmoid.get_input(0)
        gate_mm = producers[gate.name]
        assert gate_mm.type == trt.LayerType.MATRIX_MULTIPLY
        consumers = users(sigmoid.get_output(0))
        assert len(consumers) == 1
        silu, sig_index = consumers[0]
        product(silu)
        assert silu.get_input(1-sig_index).name == gate.name
        consumers = users(silu.get_output(0))
        assert len(consumers) == 1
        mul, silu_index = consumers[0]
        product(mul)
        up = mul.get_input(1-silu_index)
        up_mm = producers[up.name]
        assert up_mm.type == trt.LayerType.MATRIX_MULTIPLY
        for mm in [gate_mm, up_mm]:
            mm.__class__ = trt.IMatrixMultiplyLayer
            assert mm.op0 == mm.op1 == trt.MatrixOperation.NONE
            assert tuple(mm.get_output(0).shape) == (1,50,2048)
            assert mm.get_output(0).dtype == trt.float16
        x = gate_mm.get_input(0)
        assert x.name == up_mm.get_input(0).name and tuple(x.shape) == (1,50,720)
        assert x.dtype == trt.float16
        assert {l.name for l,i in users(gate)} == {sigmoid.name, silu.name}
        assert [l.name for l,i in users(up)] == [mul.name]
        packed = np.ascontiguousarray(np.stack([constant(gate_mm.get_input(1)), constant(up_mm.get_input(1))], axis=-1).reshape(720,4096))
        weight_buffers.append(packed)
        w = net.add_constant(packed.shape, packed).get_output(0)
        custom = net.add_plugin(trtp.op.nano_vla.expert_gated_projection(x,w),aot=True)
        custom.name = 'triton_ffn_'+sigmoid.name
        out, old = custom.get_output(0), mul.get_output(0)
        assert tuple(out.shape) == tuple(old.shape) and out.dtype == old.dtype
        downstream = users(old)
        for l,i in downstream:
            l.set_input(i,out)
        patches.append(dict(sigmoid=sigmoid.name,input=x.name,replaced=old.name,
                            consumers=[l.name for l,i in downstream],packed_weight_sha256=hashlib.sha256(packed.tobytes()).hexdigest()))

config = builder.create_builder_config()
config.builder_optimization_level = args.opt_level
accumulated = configure_accumulate(net, config, args.accumulate)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,512<<20)
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
timing = base/'timing.cache'
config.set_timing_cache(config.create_timing_cache(timing.read_bytes() if timing.exists() else b''),ignore_mismatch=False)
started = time.perf_counter()
plan = builder.build_serialized_network(net,config)
assert plan is not None
(dest/'decode.engine').write_bytes(plan)
(dest/'timing.cache').write_bytes(config.get_timing_cache().serialize())
manifest = dict(base_manifest)
manifest.update(mode='expert-'+args.mode,base_cache=str(base),vision_candidate=base_manifest,
                expert_build_s=time.perf_counter()-started,expert_opt_level=args.opt_level,expert_accumulate=args.accumulate,expert_fp32_accumulated_matmuls=accumulated,expert_patches=patches,
                additional_source_sha256={**base_manifest.get('additional_source_sha256', {}), source.name:sha256(source)},
                all_engine_sha256={p.name:sha256(p) for p in dest.glob('*.engine')},
                expert_kernel_sha256=sha256(Path(__file__).with_name('ffn_kernel.py')),
                expert_plugin_sha256=sha256(Path(__file__).with_name('ffn_plugin.py')),
                expert_precision='FP32 accumulation, HALF projection/sigmoid/SiLU/product rounding')
(dest/'candidate.json').write_text(json.dumps(manifest,indent=2))
print('BUILT',args.mode,'patches',len(patches),'seconds',manifest['expert_build_s'],flush=True)
