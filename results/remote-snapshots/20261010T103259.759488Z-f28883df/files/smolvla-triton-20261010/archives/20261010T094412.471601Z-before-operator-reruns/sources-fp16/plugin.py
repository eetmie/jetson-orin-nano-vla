"""AOT Triton softmax plugin; compiler dependencies are imported only at build."""
from typing import Tuple, Union
import tensorrt as trt
import tensorrt.plugin as trtp


@trtp.register('nano_vla::row_softmax')
def describe(inp: trtp.TensorDesc) -> trtp.TensorDesc:
    return inp.like()


@trtp.autotune('nano_vla::row_softmax')
def tune(inp: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]):
    return [trtp.AutoTuneCombination('FP32|FP16, FP32|FP16')]


@trtp.aot_impl('nano_vla::row_softmax')
def compile_kernel(inp: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc],
                   tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    import triton
    from aot_kernel import softmax_aot
    dtype='fp32' if inp.dtype==trt.float32 else 'fp16'
    src=triton.compiler.ASTSource(softmax_aot,
        signature={'X':f'*{dtype}','n_cols':'i32','row_stride':'i32','Y':f'*{dtype}'},
        constexprs={'BLOCK_N':1024})
    compiled=triton.compile(src,options={'num_warps':4})
    launch=trtp.KernelLaunchParams()
    cols=inp.shape_expr[inp.ndim-1]
    launch.grid_x=inp.shape_expr.numel()//cols
    launch.block_x=compiled.metadata.num_warps*32
    launch.shared_mem=compiled.metadata.shared
    extras=trtp.SymIntExprs.from_tuple([trtp.SymInt32(cols),trtp.SymInt32(cols)])
    return compiled.metadata.name.encode(),compiled.asm['ptx'].encode(),launch,extras
