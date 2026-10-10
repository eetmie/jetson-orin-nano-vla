"""Build an isolated SmolVLA vision candidate with twelve AOT softmax plugins.

FP32 replaces only Softmax. FP16 fuses the surrounding FP16 -> FP32 -> Softmax
-> FP16 casts, while preserving FP32 math and the existing NaN-to-zero handling.
The baseline bundle and cache are read only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import tensorrt as trt
import tensorrt.plugin as trtp
import plugin

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['fp32','fp16'],required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
source=Path.home()/'bundles/smolvla-base-split/smolvlm_vision.onnx'
base=Path.home()/'.cache/jetson-orin-nano-vla/smolvla-base-trt'
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


for soft in softmaxes:
    inp,old=soft.get_input(0),soft.get_output(0)
    assert tuple(inp.shape)==(1,12,1024,1024) and inp.dtype==trt.float32
    if a.mode=='fp16':
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
    users=consumers(old)
    custom=net.add_plugin(trtp.op.nano_vla.row_softmax(inp),aot=True)
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
              accumulation='FP32',warps=4,baseline_cache=str(base))
(a.out/'candidate.json').write_text(json.dumps(manifest,indent=2))
print('BUILT',a.mode,manifest['build_s'],flush=True)
