"""AOT TensorRT plugin for EVO1's action_output pool GEMV; Triton is imported only at build."""
from typing import Tuple, Union
import tensorrt as trt
import tensorrt.plugin as trtp

N, K = 896, 44800
TILE = dict(BLOCK_N=2, BLOCK_K=2048, num_warps=4)      # swept on the board: ~100 GB/s


@trtp.register('nano_vla::gemv_bias')
def describe(x: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc) -> trtp.TensorDesc:
    out = x.like()
    out.shape_expr[-1] = w.shape_expr[0]
    return out


@trtp.autotune('nano_vla::gemv_bias')
def tune(x: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]):
    return [trtp.AutoTuneCombination('FP16, FP16, FP16, FP16')]


@trtp.aot_impl('nano_vla::gemv_bias')
def compile_gemv(x: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc],
                 tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    import triton
    from kernels import gemv_bias_aot
    src = triton.compiler.ASTSource(gemv_bias_aot,
        signature={'X': '*fp16', 'W': '*fp16', 'B': '*fp16', 'Y': '*fp16'},
        constexprs={'N': N, 'K': K, 'BLOCK_N': TILE['BLOCK_N'], 'BLOCK_K': TILE['BLOCK_K']},
        attrs={(i,): [['tt.divisibility', 16]] for i in range(4)})
    k = triton.compile(src, options={'num_warps': TILE['num_warps']})
    assert getattr(k.metadata, 'global_scratch_size', 0) == 0
    launch = trtp.KernelLaunchParams()
    launch.grid_x = N//TILE['BLOCK_N']
    launch.block_x = k.metadata.num_warps*32
    launch.shared_mem = k.metadata.shared
    return k.metadata.name.encode(), k.asm['ptx'].encode(), launch, trtp.SymIntExprs.from_tuple([])
