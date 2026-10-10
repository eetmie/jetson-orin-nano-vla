"""AOT TensorRT plugin for π0.5's action-expert attention; Triton is imported only at build."""
from typing import Tuple, Union
import tensorrt as trt
import tensorrt.plugin as trtp

HEADS, DIM, SCALE = 8, 256, 0.0625
N_Q, N_KV = 10, 531                 # compact bundle: 10 action rows, 521 prefix + 10 keys
# Swept on the board; must stay within the 48 KiB of shared memory a kernel gets without
# an opt-in, which TensorRT's AOT launcher does not do (2 stages needed 78 KiB and failed).
TILE = dict(BLOCK_M=16, BLOCK_N=64, num_warps=8, num_stages=1)


@trtp.register('nano_vla::pi05_action_attention')
def describe(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc,
             mask: trtp.TensorDesc) -> trtp.TensorDesc:
    out = q.like()                   # [1,H,N_Q,D] -> [1,N_Q,H,D], the memory of [1,N_Q,H*D]
    out.shape_expr[1] = q.shape_expr[2]
    out.shape_expr[2] = q.shape_expr[1]
    return out


@trtp.autotune('nano_vla::pi05_action_attention')
def tune(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc, mask: trtp.TensorDesc,
         outputs: Tuple[trtp.TensorDesc]):
    return [trtp.AutoTuneCombination('FP16, FP16, FP16, FP32, FP16')]


@trtp.aot_impl('nano_vla::pi05_action_attention')
def compile_attention(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc, mask: trtp.TensorDesc,
                      outputs: Tuple[trtp.TensorDesc], tactic: int
                      ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    import triton
    from kernels import mqa_attention_aot
    src = triton.compiler.ASTSource(mqa_attention_aot,
        signature={'Q': '*fp16', 'K': '*fp16', 'V': '*fp16', 'Mask': '*fp32', 'O': '*fp16'},
        constexprs={'N_Q': N_Q, 'N_KV': N_KV, 'H': HEADS, 'D': DIM, 'SCALE': SCALE,
                    'BLOCK_M': TILE['BLOCK_M'], 'BLOCK_N': TILE['BLOCK_N']},
        attrs={(i,): [['tt.divisibility', 16]] for i in range(5)})
    k_ = triton.compile(src, options={'num_warps': TILE['num_warps'], 'num_stages': TILE['num_stages']})
    assert getattr(k_.metadata, 'global_scratch_size', 0) == 0
    assert k_.metadata.shared <= 48*1024, f'{k_.metadata.shared} B of shared memory: over the 48 KiB AOT limit'
    launch = trtp.KernelLaunchParams()
    launch.grid_x = (N_Q + TILE['BLOCK_M'] - 1)//TILE['BLOCK_M']
    launch.grid_y = HEADS
    launch.block_x = k_.metadata.num_warps*32
    launch.shared_mem = k_.metadata.shared
    return k_.metadata.name.encode(), k_.asm['ptx'].encode(), launch, trtp.SymIntExprs.from_tuple([])
