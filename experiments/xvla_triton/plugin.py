"""AOT TensorRT plugins for the X-VLA denoiser; Triton is imported only at build."""
from typing import Tuple, Union
import tensorrt as trt
import tensorrt.plugin as trtp

SCALE = 0.353515625            # HALF sqrt(1/sqrt(64)); build_denoise.py checks the graph's value
SEQ, HEADS, DIM, ACTIONS = 262, 16, 64, 30
TILE = dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2)   # swept on the board


def _compile(n_q, rows_input=False):
    import triton
    from kernels import xvla_attention_aot, xvla_attention_rows_aot
    signature = {'QKV': '*fp16', 'ROWS': '*fp16', 'O': '*fp16'} if rows_input else {'QKV': '*fp16', 'O': '*fp16'}
    src = triton.compiler.ASTSource(xvla_attention_rows_aot if rows_input else xvla_attention_aot,
        signature=signature,
        constexprs={'N_Q': n_q, 'N_KV': SEQ, 'H': HEADS, 'SCALE': SCALE, 'BLOCK_M': TILE['BLOCK_M'],
                    'BLOCK_N': TILE['BLOCK_N'], 'D': DIM},
        attrs={(i,): [['tt.divisibility', 16]] for i in range(len(signature))})
    k = triton.compile(src, options={'num_warps': TILE['num_warps'], 'num_stages': TILE['num_stages']})
    assert getattr(k.metadata, 'global_scratch_size', 0) == 0
    launch = trtp.KernelLaunchParams()
    launch.grid_x = (n_q + TILE['BLOCK_M'] - 1)//TILE['BLOCK_M']
    launch.grid_y = HEADS
    launch.block_x = k.metadata.num_warps*32
    launch.shared_mem = k.metadata.shared
    return k.metadata.name.encode(), k.asm['ptx'].encode(), launch, trtp.SymIntExprs.from_tuple([])


@trtp.register('nano_vla::xvla_attention')
def describe(qkv: trtp.TensorDesc) -> trtp.TensorDesc:
    out = qkv.like()
    out.shape_expr[-1] = qkv.shape_expr[-1]//3
    return out


@trtp.autotune('nano_vla::xvla_attention')
def tune(qkv: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]):
    return [trtp.AutoTuneCombination('FP16, FP16')]


@trtp.aot_impl('nano_vla::xvla_attention')
def compile_full(qkv: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int
                 ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    return _compile(SEQ)


@trtp.register('nano_vla::xvla_attention_actions')
def describe_actions(qkv: trtp.TensorDesc, rows: trtp.TensorDesc) -> trtp.TensorDesc:
    # rows is the [1,ACTIONS,H*D] residual slice: it only supplies the output shape.
    return rows.like()


@trtp.autotune('nano_vla::xvla_attention_actions')
def tune_actions(qkv: trtp.TensorDesc, rows: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]):
    return [trtp.AutoTuneCombination('FP16, FP16, FP16')]


@trtp.aot_impl('nano_vla::xvla_attention_actions')
def compile_actions(qkv: trtp.TensorDesc, rows: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc],
                    tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    return _compile(ACTIONS, rows_input=True)






# DaViT's four stages (views, pixels, width, channels). One op per shape: plugin attributes
# are unusable here (scalar ints go through int(array), which numpy 2 rejects, and NDArray
# annotations fail TensorRT's registration check on Python 3.12).
DWCONV_SHAPES = [(3, 3136, 56, 256), (3, 784, 28, 512), (3, 196, 14, 1024), (3, 49, 7, 2048)]


def dwconv_op(views, n_pix, width, channels):
    return f'dwconv_tokens_{views}x{n_pix}x{channels}'


def _register_dwconv(views, n_pix, width, channels):
    name = 'nano_vla::' + dwconv_op(views, n_pix, width, channels)

    @trtp.register(name)
    def describe(x: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc) -> trtp.TensorDesc:
        return x.like()

    @trtp.autotune(name)
    def tune(x: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]):
        return [trtp.AutoTuneCombination('FP16, FP16, FP16, FP16')]

    @trtp.aot_impl(name)
    def compile_dwconv(x: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc],
                       tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
        import triton
        from kernels import dwconv_tokens_aot
        # Swept per stage shape on the board (bench_dwconv.py).
        t = dict(BLOCK_P=16, BLOCK_C=128, num_warps=4) if n_pix < 1000 and channels in (512, 1024) \
            else dict(BLOCK_P=64, BLOCK_C=64, num_warps=4)
        src = triton.compiler.ASTSource(dwconv_tokens_aot,
            signature={'X': '*fp16', 'W': '*fp16', 'B': '*fp16', 'Y': '*fp16'},
            constexprs={'N_PIX': n_pix, 'WIDTH': width, 'C': channels, 'BLOCK_P': t['BLOCK_P'], 'BLOCK_C': t['BLOCK_C']},
            attrs={(i,): [['tt.divisibility', 16]] for i in range(4)})
        k = triton.compile(src, options={'num_warps': t['num_warps']})
        assert getattr(k.metadata, 'global_scratch_size', 0) == 0 and k.metadata.shared <= 48*1024
        launch = trtp.KernelLaunchParams()
        launch.grid_x = (n_pix + t['BLOCK_P'] - 1)//t['BLOCK_P']
        launch.grid_y = channels//t['BLOCK_C']
        launch.grid_z = views
        launch.block_x = k.metadata.num_warps*32
        launch.shared_mem = k.metadata.shared
        return k.metadata.name.encode(), k.asm['ptx'].encode(), launch, trtp.SymIntExprs.from_tuple([])


for _shape in DWCONV_SHAPES:
    _register_dwconv(*_shape)
