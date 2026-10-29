"""Fixed-shape GR00T TensorRT AOT registrations; Triton is imported only while building."""
from typing import Tuple, Union
import tensorrt.plugin as trtp

SPECS = {
    'n16_vision': dict(B=3,NQ=324,NK=324,H=16,D=72,QK_SCALE=1.,SCORE_SCALE=0.11785888671875,PACKED_KV=False,HAS_BIAS=False,BM=128,BN=32,SCORE_FP32=True),
    'n17_vision': dict(B=3,NQ=256,NK=256,H=16,D=64,QK_SCALE=1.,SCORE_SCALE=.125,PACKED_KV=False,HAS_BIAS=False,BM=128,BN=32,SCORE_FP32=True),
    'n16_cross': dict(B=1,NQ=51,NK=320,H=32,D=48,QK_SCALE=.3798828125,SCORE_SCALE=1.,PACKED_KV=True,HAS_BIAS=True,BM=64,BN=32),
    'n17_cross': dict(B=1,NQ=41,NK=448,H=32,D=48,QK_SCALE=.3798828125,SCORE_SCALE=1.,PACKED_KV=True,HAS_BIAS=True,BM=64,BN=64),
    'n16_self': dict(B=1,NQ=51,NK=51,H=32,D=48,QK_SCALE=.3798828125,SCORE_SCALE=1.,PACKED_KV=False,HAS_BIAS=False,BM=64,BN=64),
    'n17_self': dict(B=1,NQ=41,NK=41,H=32,D=48,QK_SCALE=.3798828125,SCORE_SCALE=1.,PACKED_KV=False,HAS_BIAS=False,BM=64,BN=64),
}


def gathered_spec(family, nk):
    """Cross-attention on K/V gathered to the rows the block can see (export default since
    the K/V row gather): packed FP16 [K|V], no mask. NK depends on the baked prompt."""
    nq = {'n16': 51, 'n17': 41}[family]
    return dict(B=1, NQ=nq, NK=nk, H=32, D=48, QK_SCALE=.3798828125, SCORE_SCALE=1.,
                PACKED_KV=True, HAS_BIAS=False, KV_HALF=True, BM=64, BN=64 if nk <= 64 else 32,
                # one key tile: the exact (normalize-then-round) order costs nothing here
                EXACT=nk <= 64)


def register_gathered(family, nk):
    key = f'{family}_cross{nk}'
    if key not in SPECS:
        SPECS[key] = gathered_spec(family, nk)
        SPECS[key]['V_PACKED'] = False
        for exact in [False, True]:
            _register(key, SPECS[key], exact)
    return key


def op_name(shape, two_pass=False):
    return 'groot_'+shape+('_exact' if two_pass else '_online')


def _register(shape, spec, two_pass):
    name='nano_vla::'+op_name(shape,two_pass)
    @trtp.register(name)
    def describe(q:trtp.TensorDesc,k:trtp.TensorDesc,v:trtp.TensorDesc,bias:trtp.TensorDesc)->trtp.TensorDesc:
        return q.like()
    @trtp.autotune(name)
    def tune(q:trtp.TensorDesc,k:trtp.TensorDesc,v:trtp.TensorDesc,bias:trtp.TensorDesc,outputs:Tuple[trtp.TensorDesc]):
        kv='FP32' if spec['PACKED_KV'] and not spec.get('KV_HALF') else 'FP16'
        formats=f"FP16, {kv}, FP16, {'FP32' if spec['HAS_BIAS'] else 'FP16'}, FP16"
        return [trtp.AutoTuneCombination(formats)]
    @trtp.aot_impl(name)
    def compile_kernel(q:trtp.TensorDesc,k:trtp.TensorDesc,v:trtp.TensorDesc,bias:trtp.TensorDesc,
                       outputs:Tuple[trtp.TensorDesc],tactic:int
                       )->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymExprs]:
        import triton
        from bench.vendor.triton_compile import compile_sm87
        from kernels import attention_aot, attention72_aot
        signature={'Q':'*fp16','K':'*fp32' if spec['PACKED_KV'] and not spec.get('KV_HALF') else '*fp16',
                   'V':'*fp16','Bias':'*fp32' if spec['HAS_BIAS'] else '*fp16','O':'*fp16'}
        src=triton.compiler.ASTSource(attention72_aot if spec['D']==72 else attention_aot,signature=signature,
            constexprs={**{k:v for k,v in spec.items() if k not in ('KV_HALF','EXACT')},'KD':triton.next_power_of_2(spec['D']),'TWO_PASS':two_pass},
            attrs={(i,):[['tt.divisibility',16]] for i in range(len(signature))})
        kernel=compile_sm87(src,options={'num_warps':4,'num_stages':2})
        assert not getattr(kernel.metadata,'global_scratch_size',0)
        assert kernel.metadata.shared<=49152, 'AOT launcher cannot opt into larger shared memory'
        launch=trtp.KernelLaunchParams()
        launch.grid_x=(spec['NQ']+spec['BM']-1)//spec['BM']
        launch.grid_y=spec['H'];launch.grid_z=spec['B']
        launch.block_x=kernel.metadata.num_warps*32;launch.shared_mem=kernel.metadata.shared
        return kernel.metadata.name.encode(),kernel.asm['ptx'].encode(),launch,trtp.SymIntExprs.from_tuple([])

for _shape,_spec in SPECS.items():
    _spec['V_PACKED']=_shape=='n17_vision'
    for _exact in [False,True]:_register(_shape,_spec,_exact)


def register_variant(key, **shape):
    """The base spec with another camera count (B) or prompt length (NK), as a custom
    fine-tune exports it; the base shapes keep their names."""
    if all(SPECS[key].get(k) == v for k, v in shape.items()):
        return key
    new = key + "".join(f"_{k}{v}" for k, v in shape.items())
    if new not in SPECS:
        SPECS[new] = {**SPECS[key], **shape}
        for exact in [False, True]:
            _register(new, SPECS[new], exact)
    return new


def register_causal(family, S, H, G, D, scale, strides, two_pass=True):
    """Causal GQA LLM attention (kernels.causal_aot) for one fixed shape and layout."""
    key = f'{family}_causal{S}f32_' + '_'.join(str(s) for s in strides)
    if key in SPECS:
        return key
    spec = dict(S=S, H=H, G=G, D=D, SCALE=scale, SQH=strides[0], SQS=strides[1],
                SKH=strides[2], SKS=strides[3], SVH=strides[4], SVS=strides[5],
                TWO_PASS=two_pass, BM=64, BN=32, SCORE_FP32=True)
    SPECS[key] = spec
    name = 'nano_vla::' + op_name(key, two_pass)

    @trtp.register(name)
    def describe(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc,
                 bias: trtp.TensorDesc) -> trtp.TensorDesc:
        return trtp.from_shape_expr((1, S, H * D), dtype=q.dtype)

    @trtp.autotune(name)
    def tune(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc, bias: trtp.TensorDesc,
             outputs: Tuple[trtp.TensorDesc]):
        return [trtp.AutoTuneCombination('FP16, FP16, FP16, FP16, FP16')]

    @trtp.aot_impl(name)
    def compile_kernel(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc,
                       bias: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int
                       ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams,
                                  trtp.SymExprs]:
        import triton
        from bench.vendor.triton_compile import compile_sm87
        from kernels import causal_aot
        signature = {'Q': '*fp16', 'K': '*fp16', 'V': '*fp16', 'Bias': '*fp16', 'O': '*fp16'}
        src = triton.compiler.ASTSource(causal_aot, signature=signature, constexprs=dict(spec),
                                        attrs={(i,): [['tt.divisibility', 16]] for i in range(5)})
        kernel = compile_sm87(src, options={'num_warps': 4, 'num_stages': 2})
        assert not getattr(kernel.metadata, 'global_scratch_size', 0)
        assert kernel.metadata.shared <= 49152, 'AOT launcher cannot opt into larger shared memory'
        launch = trtp.KernelLaunchParams()
        launch.grid_x = (S + spec['BM'] - 1) // spec['BM']
        launch.grid_y = H
        launch.grid_z = 1
        launch.block_x = kernel.metadata.num_warps * 32
        launch.shared_mem = kernel.metadata.shared
        return kernel.metadata.name.encode(), kernel.asm['ptx'].encode(), launch, trtp.SymIntExprs.from_tuple([])
    return key
