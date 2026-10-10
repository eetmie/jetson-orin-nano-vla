"""Fixed expert gated-projection plugin; compiler dependencies only at build."""
from typing import Tuple, Union
import tensorrt as trt
import tensorrt.plugin as trtp

@trtp.register('nano_vla::expert_gated_projection')
def describe(x: trtp.TensorDesc, weight: trtp.TensorDesc) -> trtp.TensorDesc:
    out = x.like()
    out.shape_expr[-1] = weight.shape_expr[1]//2
    return out

@trtp.autotune('nano_vla::expert_gated_projection')
def tune(x: trtp.TensorDesc, weight: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]):
    return [trtp.AutoTuneCombination('FP16, FP16, FP16')]

@trtp.aot_impl('nano_vla::expert_gated_projection')
def compile_kernel(x: trtp.TensorDesc, weight: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc],
                   tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    import triton
    from ffn_kernel import packed_gated_projection
    source = triton.compiler.ASTSource(packed_gated_projection,
        signature={'X':'*fp16','Weight':'*fp16','Y':'*fp16'},
        constexprs={'M':50,'N':2048,'K':720,'BM':64,'BN':128,'BK':32},
        attrs={(i,):[['tt.divisibility',16]] for i in range(3)})
    kernel = triton.compile(source,options={'num_warps':4,'num_stages':2,'enable_fp_fusion':False})
    assert getattr(kernel.metadata,'global_scratch_size',0) == 0
    launch = trtp.KernelLaunchParams()
    launch.grid_x = 1
    launch.grid_y = 16
    launch.block_x = 128
    launch.shared_mem = kernel.metadata.shared
    return kernel.metadata.name.encode(), kernel.asm['ptx'].encode(), launch, trtp.SymIntExprs.from_tuple([])
