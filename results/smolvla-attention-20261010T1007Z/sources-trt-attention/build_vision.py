"""Build an isolated SmolVLA vision candidate with twelve AOT softmax plugins.

FP32 replaces only Softmax. FP16 also covers surrounding casts. Masked covers
Add/Cast/Softmax/Cast/IsNaN/Where, preserving FP16 Add rounding and FP32 softmax
math. Baseline rebuilds the original vision graph as a control. All modes use a
new cache; the baseline bundle and cache are read only.
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

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['baseline','fp32','fp16','masked','trt-attention'],required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
source=Path.home()/'bundles/smolvla-base-split/smolvlm_vision.onnx'
base=Path.home()/'.cache/jetson-orin-nano-vla/smolvla-base-trt'
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


for soft in (softmaxes if a.mode!='baseline' else []):
    inp,old=soft.get_input(0),soft.get_output(0)
    assert tuple(inp.shape)==(1,12,1024,1024) and inp.dtype==trt.float32
    if a.mode in ['fp16','masked','trt-attention']:
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
    if a.mode in ['masked','trt-attention']:
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
            k_shuffle=net.add_shuffle(kt)
            k_shuffle.second_transpose=trt.Permutation((0,1,3,2))
            k=k_shuffle.get_output(0)
            custom=net.add_attention(q,k,v,trt.AttentionNormalizationOp.SOFTMAX,False)
            custom.mask=mask
            custom.decomposable=False  # A failed fusion must fail this experiment.
            old=pv.get_output(0)
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
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,512<<20)
config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
cache=base/'timing.cache'
config.set_timing_cache(config.create_timing_cache(cache.read_bytes() if cache.exists() else b''),ignore_mismatch=False)
started=time.time()
plan=builder.build_serialized_network(net,config)
assert plan is not None, 'candidate engine build failed'
(a.out/'vision.engine').write_bytes(plan)
(a.out/'timing.cache').write_bytes(config.get_timing_cache().serialize())
manifest=dict(mode=a.mode,build_s=time.time()-started,tensorrt=trt.__version__,
              source_onnx_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              engine_sha256=hashlib.sha256(bytes(plan)).hexdigest(),patches=patches,
              kernel_sha256=hashlib.sha256((Path(__file__).parent/'aot_kernel.py').read_bytes()).hexdigest(),
              plugin_sha256=hashlib.sha256((Path(__file__).parent/'plugin.py').read_bytes()).hexdigest(),
              accumulation=('TensorRT IAttention internal precision' if a.mode=='trt-attention' else 'FP32'),
              implementation=('TensorRT IAttention' if a.mode=='trt-attention' else 'Triton AOT' if a.mode!='baseline' else 'TensorRT original graph'),
              warps=(None if a.mode in ['baseline','trt-attention'] else 4),baseline_cache=str(base))
(a.out/'candidate.json').write_text(json.dumps(manifest,indent=2))
print('BUILT',a.mode,manifest['build_s'],flush=True)
