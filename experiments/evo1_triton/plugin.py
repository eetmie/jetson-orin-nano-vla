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
    from bench.vendor.triton_compile import compile_sm87
    from kernels import gemv_bias_aot
    src = triton.compiler.ASTSource(gemv_bias_aot,
        signature={'X': '*fp16', 'W': '*fp16', 'B': '*fp16', 'Y': '*fp16'},
        constexprs={'N': N, 'K': K, 'BLOCK_N': TILE['BLOCK_N'], 'BLOCK_K': TILE['BLOCK_K']},
        attrs={(i,): [['tt.divisibility', 16]] for i in range(4)})
    k = compile_sm87(src, options={'num_warps': TILE['num_warps']})
    assert getattr(k.metadata, 'global_scratch_size', 0) == 0
    launch = trtp.KernelLaunchParams()
    launch.grid_x = N//TILE['BLOCK_N']
    launch.block_x = k.metadata.num_warps*32
    launch.shared_mem = k.metadata.shared
    return k.metadata.name.encode(), k.asm['ptx'].encode(), launch, trtp.SymIntExprs.from_tuple([])


STEP_TILE = dict(BM=32, BN=64, num_warps=4, num_stages=1)   # swept on the board (bench_step.py)
STEP_SPECS = {}


def register_step(NQ, NK, H, D, scale, strides, score_fp32=True, two_pass=False):
    """Action-step cross-attention (kernels.step_attention_aot) for one shape and layout."""
    key = f'evo1_step_{NQ}_{NK}_{H}_{D}_' + '_'.join(map(str, strides)) + ('_f32' if score_fp32 else '_g') + ('_x' if two_pass else '_o')
    if key in STEP_SPECS:
        return key
    spec = dict(NQ=NQ, NK=NK, H=H, D=D, KD=1 << (D - 1).bit_length(), SCALE=scale,
                SQH=strides[0], SQS=strides[1], SKH=strides[2], SKS=strides[3], SVH=strides[4], SVS=strides[5],
                SCORE_FP32=score_fp32, TWO_PASS=two_pass, BM=STEP_TILE['BM'], BN=STEP_TILE['BN'])
    STEP_SPECS[key] = spec
    name = 'nano_vla::' + key

    @trtp.register(name)
    def describe(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc, m: trtp.TensorDesc) -> trtp.TensorDesc:
        return trtp.from_shape_expr((1, NQ, H * D), dtype=q.dtype)

    @trtp.autotune(name)
    def tune(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc, m: trtp.TensorDesc,
             outputs: Tuple[trtp.TensorDesc]):
        return [trtp.AutoTuneCombination('FP16, FP16, FP16, FP16, FP16')]

    @trtp.aot_impl(name)
    def compile_step(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc, m: trtp.TensorDesc,
                     outputs: Tuple[trtp.TensorDesc], tactic: int
                     ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
        import triton
        from bench.vendor.triton_compile import compile_sm87
        from kernels import step_attention_aot
        src = triton.compiler.ASTSource(step_attention_aot,
            signature={'Q': '*fp16', 'K': '*fp16', 'V': '*fp16', 'Mask': '*fp16', 'O': '*fp16'},
            constexprs=dict(spec), attrs={(i,): [['tt.divisibility', 16]] for i in range(5)})
        k_ = compile_sm87(src, options={'num_warps': STEP_TILE['num_warps'], 'num_stages': STEP_TILE['num_stages']})
        assert getattr(k_.metadata, 'global_scratch_size', 0) == 0
        assert k_.metadata.shared <= 49152, 'AOT launcher cannot opt into larger shared memory'
        launch = trtp.KernelLaunchParams()
        launch.grid_x = (NQ + spec['BM'] - 1) // spec['BM']
        launch.grid_y = H
        launch.block_x = k_.metadata.num_warps * 32
        launch.shared_mem = k_.metadata.shared
        return k_.metadata.name.encode(), k_.asm['ptx'].encode(), launch, trtp.SymIntExprs.from_tuple([])
    return key
