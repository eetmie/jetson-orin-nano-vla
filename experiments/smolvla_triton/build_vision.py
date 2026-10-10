"""Build an isolated SmolVLA vision candidate with twelve AOT softmax plugins.

FP32 replaces only Softmax. FP16 also covers surrounding casts. Masked covers
Add/Cast/Softmax/Cast/IsNaN/Where, preserving FP16 Add rounding and FP32 softmax
math. Baseline rebuilds the original vision graph as a control. All modes use a
new cache; the baseline bundle and cache are read only. triton-attention-native
takes Q/K/V before their transposes and replaces the output transpose too; with
--base-cache it rebuilds only vision.engine on top of a verified candidate cache.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import time

import tensorrt as trt
import tensorrt.plugin as trtp
import plugin
from candidate_cache import sha256, verify_candidate
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.vendor.groot_trt import configure_accumulate

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['baseline','fp32','fp16','masked','trt-attention','triton-attention','triton-attention-aligned','triton-attention-native','triton-attention-flat','triton-attention-qkv'],required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--base-cache',type=Path,help='cache whose other engines are kept (verified if it is a candidate)')
p.add_argument('--bundle',type=Path,default=Path.home()/'bundles/smolvla-base-split')
p.add_argument('--accumulate', choices=['fp32', 'auto'], default='fp32', help='as bench trt-split --accumulate')
a=p.parse_args()
source=a.bundle.expanduser().resolve()/'smolvlm_vision.onnx'
base=Path.home()/'.cache/jetson-orin-nano-vla/smolvla-base-trt'
base_manifest=None
if a.base_cache:
    base=a.base_cache.expanduser().resolve()
    if (base/'candidate.json').exists():
        base_manifest=verify_candidate(base,source.parent)
a.out=a.out.expanduser().resolve()
assert a.out!=base.resolve(), 'Use a separate experimental cache'
if (a.out/'vision.engine').exists():
    raise FileExistsError(f'{a.out}/vision.engine already exists; use a new cache per build')
a.out.mkdir(parents=True,exist_ok=True)
for path in base.glob('*.engine'):
    if path.name!='vision.engine':shutil.copy2(path,a.out/path.name)
logger=trt.Logger(trt.Logger.WARNING)
builder=trt.Builder(logger)
net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
parser=trt.OnnxParser(net,logger)
assert parser.parse_from_file(str(source)), [str(parser.get_error(i)) for i in range(parser.num_errors)]
original=[net.get_layer(i) for i in range(net.num_layers)]
producers={l.get_output(j).name:l for l in original for j in range(l.num_outputs)}
softmaxes=[l for l in original if l.type==trt.LayerType.SOFTMAX]
assert len(softmaxes)==12
patches=[]


def consumers(t):
    return [(l,j) for l in original for j in range(l.num_inputs)
            if l.get_input(j) is not None and l.get_input(j).name==t.name]


NATIVE=(1,1024,12,64)
network_inputs={net.get_input(i).name for i in range(net.num_inputs)}


def emulate(layer,arr):
    """Apply one parsed Reshape/Transpose shuffle to an index array."""
    layer.__class__=trt.IShuffleLayer
    assert layer.num_inputs==1 or layer.get_input(1) is None, f'{layer.name}: dynamic reshape'
    arr=arr.transpose(tuple(layer.first_transpose)[:arr.ndim])
    try:
        dims=tuple(layer.reshape_dims)
    except ValueError:  # nbDims -1: no reshape set
        dims=()
    if dims:
        if layer.zero_is_placeholder:
            dims=tuple(arr.shape[i] if d==0 else d for i,d in enumerate(dims))
        arr=arr.reshape(dims)
    return arr.transpose(tuple(layer.second_transpose)[:arr.ndim])


def constant_value(t):
    layer=producers[t.name]
    while layer.type in [trt.LayerType.SHUFFLE,trt.LayerType.CAST]:
        layer=producers[layer.get_input(0).name]
    assert layer.type==trt.LayerType.CONSTANT, layer.name
    layer.__class__=trt.IConstantLayer
    return np.asarray(layer.weights)


def native(t,perm):
    """Walk t back to the [1,S,H,D] projection view; check the shuffles equal perm."""
    shuffles,scale=[],None
    while tuple(t.shape)!=NATIVE:
        layer=producers[t.name]
        if layer.type==trt.LayerType.SHUFFLE:
            shuffles.append(layer)
            t=layer.get_input(0)
            continue
        assert layer.type==trt.LayerType.ELEMENTWISE and scale is None, layer.name
        layer.__class__=trt.IElementWiseLayer
        assert layer.op==trt.ElementWiseOperation.PROD
        x,c=layer.get_input(0),layer.get_input(1)
        try:
            value=constant_value(c)
        except AssertionError:
            x,c=c,x
            value=constant_value(c)
        assert value.size==1 and value.dtype==np.float16, value
        scale=float(value.reshape(-1)[0])
        t=x
    idx=np.arange(np.prod(NATIVE)).reshape(NATIVE)
    arr=idx
    for layer in reversed(shuffles):
        arr=emulate(layer,arr)
    assert np.array_equal(arr,idx.transpose(perm)), (t.name,perm)
    return t,scale


def flat_source(t):
    """The [1,1024,768] tensor a [1,1024,12,64] view was reshaped from (pure reshape)."""
    layer=producers[t.name]
    assert layer.type==trt.LayerType.SHUFFLE, layer.name
    src=layer.get_input(0)
    assert tuple(src.shape)==(1,1024,768), tuple(src.shape)
    idx=np.arange(np.prod(NATIVE)).reshape(1,1024,768)
    assert np.array_equal(emulate(layer,idx),idx.reshape(NATIVE))
    return src


keep_alive=[]


def projection(t):
    """(x, W[768,768], b[768]) of t = MatMul(x, W) + b, all HALF constants."""
    add=producers[t.name]
    assert add.type==trt.LayerType.ELEMENTWISE, add.name
    add.__class__=trt.IElementWiseLayer
    assert add.op==trt.ElementWiseOperation.SUM
    mm,bias=add.get_input(0),add.get_input(1)
    if producers[mm.name].type!=trt.LayerType.MATRIX_MULTIPLY:
        mm,bias=bias,mm
    mm=producers[mm.name]
    assert mm.type==trt.LayerType.MATRIX_MULTIPLY, mm.name
    mm.__class__=trt.IMatrixMultiplyLayer
    assert mm.op0==mm.op1==trt.MatrixOperation.NONE
    w=constant_value(mm.get_input(1))
    b=constant_value(bias)
    assert w.dtype==b.dtype==np.float16 and w.size==768*768 and b.size==768, (w.shape,b.shape)
    return mm.get_input(0),w.reshape(768,768),b.reshape(768)


def constant_only(t):
    stack,seen=[t],set()
    while stack:
        x=stack.pop()
        if x.name in seen:
            continue
        seen.add(x.name)
        assert x.name not in network_inputs, f'mask depends on network input {x.name}'
        layer=producers.get(x.name)
        if layer is not None:
            stack+=[layer.get_input(i) for i in range(layer.num_inputs) if layer.get_input(i) is not None]
    return True


for soft in (softmaxes if a.mode!='baseline' else []):
    inp,old=soft.get_input(0),soft.get_output(0)
    assert tuple(inp.shape)==(1,12,1024,1024) and inp.dtype==trt.float32
    if a.mode in ['fp16','masked','trt-attention'] or a.mode.startswith('triton-attention'):
        pred=producers[inp.name]
        assert pred.type==trt.LayerType.CAST and pred.get_input(0).dtype==trt.float16
        inp=pred.get_input(0)
        # ONNX parser preserves shape through its softmax shim before the output cast.
        while True:
            users=consumers(old)
            assert len(users)==1, [(l.name,j) for l,j in users]
            layer,_=users[0]
            assert tuple(layer.get_output(0).shape)==tuple(inp.shape)
            if layer.type==trt.LayerType.SHUFFLE:
                old=layer.get_output(0)
            else:
                assert layer.type==trt.LayerType.CAST and layer.get_output(0).dtype==trt.float16
                old=layer.get_output(0)
                break
    if a.mode in ['masked','trt-attention'] or a.mode.startswith('triton-attention'):
        # Validate every link of the exact exported Add/Cast/Softmax/Cast/IsNaN/Where chain.
        add=producers[inp.name]
        assert add.type==trt.LayerType.ELEMENTWISE
        add.__class__=trt.IElementWiseLayer
        assert add.op==trt.ElementWiseOperation.SUM
        logits,mask=add.get_input(0),add.get_input(1)
        assert tuple(logits.shape)==(1,12,1024,1024) and tuple(mask.shape)==(1,1,1024,1024)
        assert logits.dtype==mask.dtype==trt.float16
        select=next(l for l,j in consumers(old) if l.type==trt.LayerType.SELECT)
        assert select.get_input(2).name==old.name
        nan=producers[select.get_input(0).name]
        nan.__class__=trt.IUnaryLayer
        assert nan.op==trt.UnaryOperation.ISNAN and nan.get_input(0).name==old.name
        true_value=select.get_input(1)
        const=producers[true_value.name]
        while const.type in [trt.LayerType.SHUFFLE,trt.LayerType.CAST]:
            const=producers[const.get_input(0).name]
        assert const.type==trt.LayerType.CONSTANT
        const.__class__=trt.IConstantLayer
        assert np.all(np.asarray(const.weights)==0), 'NaN replacement must be exactly zero'
        old=select.get_output(0)
        inp=logits
        if a.mode=='masked':
            custom=net.add_plugin(trtp.op.nano_vla.masked_row_softmax(logits,mask),aot=True)
        else:
            qk=producers[logits.name]
            assert qk.type==trt.LayerType.MATRIX_MULTIPLY
            qk.__class__=trt.IMatrixMultiplyLayer
            assert qk.op0==qk.op1==trt.MatrixOperation.NONE
            q,kt=qk.get_input(0),qk.get_input(1)
            assert tuple(q.shape)==(1,12,1024,64) and tuple(kt.shape)==(1,12,64,1024)
            pv_users=consumers(old)
            assert len(pv_users)==1 and pv_users[0][1]==0
            pv=pv_users[0][0]
            assert pv.type==trt.LayerType.MATRIX_MULTIPLY
            pv.__class__=trt.IMatrixMultiplyLayer
            assert pv.op0==pv.op1==trt.MatrixOperation.NONE
            v=pv.get_input(1)
            assert tuple(v.shape)==tuple(q.shape) and q.dtype==kt.dtype==v.dtype==trt.float16
            # Retain the exported HALF scaling of both Q and K. IAttention expects
            # K in BHSD order instead of the existing BMM's BHDS order.
            if a.mode=='trt-attention':
                k_shuffle=net.add_shuffle(kt)
                k_shuffle.second_transpose=trt.Permutation((0,1,3,2))
                k=k_shuffle.get_output(0)
                custom=net.add_attention(q,k,v,trt.AttentionNormalizationOp.SOFTMAX,False)
                custom.mask=mask
                custom.decomposable=False  # A failed fusion must fail this experiment.
            elif a.mode in ['triton-attention-native','triton-attention-flat','triton-attention-qkv']:
                # The mask is built from constants only (all-true for a full image),
                # so it adds exact zeros; the kernel does not read it.
                assert constant_only(mask)
                qn,q_scale=native(q,(0,2,1,3))
                kn,k_scale=native(kt,(0,2,3,1))
                vn,v_scale=native(v,(0,2,1,3))
                assert q_scale==k_scale==plugin.VISION_SCALE and v_scale is None, (q_scale,k_scale,v_scale)
                after=consumers(pv.get_output(0))
                assert len(after)==1 and after[0][0].type==trt.LayerType.SHUFFLE
                transpose=after[0][0]
                idx=np.arange(np.prod(NATIVE)).reshape(1,12,1024,64)
                assert np.array_equal(emulate(transpose,idx),idx.transpose(0,2,1,3))
                if a.mode in ['triton-attention-flat','triton-attention-qkv']:
                    # Read the [1,1024,768] projections and write what o_proj reads.
                    qn,kn,vn=[flat_source(t) for t in (qn,kn,vn)]
                    reshape=consumers(transpose.get_output(0))
                    assert len(reshape)==1 and reshape[0][0].type==trt.LayerType.SHUFFLE
                    reshape=reshape[0][0]
                    idx=np.arange(np.prod(NATIVE)).reshape(NATIVE)
                    assert np.array_equal(emulate(reshape,idx),idx.reshape(1,1024,768))
                    transpose=reshape
                    if a.mode=='triton-attention-qkv':
                        # One [768,2304] projection instead of three, as the original
                        # engine's fused QKV GEMM; the kernel reads Q|K|V columns.
                        parts=[projection(t) for t in (qn,kn,vn)]
                        assert len({x.name for x,_,_ in parts})==1
                        w=np.ascontiguousarray(np.concatenate([w for _,w,_ in parts],1)[None])
                        b=np.ascontiguousarray(np.concatenate([b for _,_,b in parts])[None,None])
                        keep_alive+=[w,b]
                        mm=net.add_matrix_multiply(parts[0][0],trt.MatrixOperation.NONE,
                                                   net.add_constant(w.shape,trt.Weights(w)).get_output(0),trt.MatrixOperation.NONE)
                        qkv=net.add_elementwise(mm.get_output(0),net.add_constant(b.shape,trt.Weights(b)).get_output(0),
                                                trt.ElementWiseOperation.SUM).get_output(0)
                        assert tuple(qkv.shape)==(1,1024,2304) and qkv.dtype==trt.float16
                        custom=net.add_plugin(trtp.op.nano_vla.vision_attention_qkv(qkv),aot=True)
                    else:
                        custom=net.add_plugin(trtp.op.nano_vla.vision_attention_flat(qn,kn,vn),aot=True)
                else:
                    custom=net.add_plugin(trtp.op.nano_vla.vision_attention_native(qn,kn,vn),aot=True)
            else:
                op=(trtp.op.nano_vla.vision_attention_aligned if a.mode.endswith('-aligned')
                    else trtp.op.nano_vla.vision_attention)
                custom=net.add_plugin(op(q,kt,v,mask),aot=True)
            old=transpose.get_output(0) if a.mode in ['triton-attention-native','triton-attention-flat','triton-attention-qkv'] else pv.get_output(0)
    else:
        custom=net.add_plugin(trtp.op.nano_vla.row_softmax(inp),aot=True)
    users=consumers(old)
    custom.name=f'triton_{a.mode}_{soft.name}'
    replacement=custom.get_output(0)
    assert tuple(replacement.shape)==tuple(old.shape) and replacement.dtype==old.dtype
    for layer,index in users:layer.set_input(index,replacement)
    patches.append(dict(softmax=soft.name,input=inp.name,replaced_output=old.name,
                        input_dtype=str(inp.dtype),output_dtype=str(old.dtype),
                        consumers=[l.name for l,_ in users]))
config=builder.create_builder_config()
config.builder_optimization_level=2
accumulated = configure_accumulate(net, config, a.accumulate)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,512<<20)
config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
cache=base/'timing.cache'
config.set_timing_cache(config.create_timing_cache(cache.read_bytes() if cache.exists() else b''),ignore_mismatch=False)
started=time.time()
plan=builder.build_serialized_network(net,config)
assert plan is not None, 'candidate engine build failed'
(a.out/'vision.engine').write_bytes(plan)
(a.out/'timing.cache').write_bytes(config.get_timing_cache().serialize())
manifest=dict(mode=a.mode,accumulate=a.accumulate,fp32_accumulated_matmuls=accumulated,build_s=time.time()-started,tensorrt=trt.__version__,
              source_onnx_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              engine_sha256=hashlib.sha256(bytes(plan)).hexdigest(),patches=patches,
              kernel_sha256=hashlib.sha256((Path(__file__).parent/'aot_kernel.py').read_bytes()).hexdigest(),
              plugin_sha256=hashlib.sha256((Path(__file__).parent/'plugin.py').read_bytes()).hexdigest(),
              accumulation=('TensorRT IAttention internal precision' if a.mode=='trt-attention' else 'FP32'),
              implementation=('TensorRT IAttention' if a.mode=='trt-attention' else 'Triton AOT' if a.mode!='baseline' else 'TensorRT original graph'),
              warps=(None if a.mode in ['baseline','trt-attention'] else 4),baseline_cache=str(base))
if a.mode.startswith('triton-attention'):
    manifest.update(attention_kernel_sha256=hashlib.sha256((Path(__file__).parent/'attention_kernel.py').read_bytes()).hexdigest(),
                    block_m=128,block_n=64,stages=2,alignment16=a.mode.endswith('-aligned'),
                    precision_note='HALF score/Add rounding retained; online probabilities round before global normalization')
if a.mode in ['triton-attention-native','triton-attention-flat','triton-attention-qkv']:
    tile=plugin.NATIVE_TILE
    manifest.update(block_m=tile['BLOCK_M'],block_n=tile['BLOCK_N'],stages=tile['num_stages'],
                    warps=tile['num_warps'],alignment16=True,layout='[1,S,H,D] projection views, no transposes',
                    mask='constant all-true, not read')
if a.base_cache and base_manifest is None:
    # A plain prebuilt cache: pin every engine and every other graph it was built from.
    manifest.update(base_cache=str(base),all_engine_sha256={p.name:sha256(p) for p in a.out.glob('*.engine')},
                    additional_source_sha256={p.name:sha256(p) for p in source.parent.glob('*.onnx') if p!=source})
elif base_manifest is not None:
    vision=manifest
    manifest=dict(base_manifest)
    manifest.update(mode=base_manifest['mode']+'+vision-'+a.mode,vision_rebuilt=vision,engine_sha256=vision['engine_sha256'],base_cache=str(base),
                    all_engine_sha256={p.name:sha256(p) for p in a.out.glob('*.engine')})
(a.out/'candidate.json').write_text(json.dumps(manifest,indent=2))
print('BUILT',a.mode,round(time.time()-started,1),flush=True)
