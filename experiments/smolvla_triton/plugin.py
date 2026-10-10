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


@trtp.register('nano_vla::vision_attention')
def describe_attention(q: trtp.TensorDesc, kt: trtp.TensorDesc, v: trtp.TensorDesc,
                       mask: trtp.TensorDesc) -> trtp.TensorDesc:
    return q.like()


@trtp.autotune('nano_vla::vision_attention')
def tune_attention(q: trtp.TensorDesc, kt: trtp.TensorDesc, v: trtp.TensorDesc,
                   mask: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]):
    return [trtp.AutoTuneCombination('FP16, FP16, FP16, FP16, FP16')]


@trtp.aot_impl('nano_vla::vision_attention')
def compile_attention(q: trtp.TensorDesc, kt: trtp.TensorDesc, v: trtp.TensorDesc,
                      mask: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int
                      ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    return _compile_attention(q, aligned=False)


def _compile_attention(q, aligned):
    import triton
    from attention_kernel import attention_aot
    src=triton.compiler.ASTSource(attention_aot,
        signature={'Q':'*fp16','Kt':'*fp16','V':'*fp16','Mask':'*fp16','n_ctx':'i32','O':'*fp16'},
        constexprs={'BLOCK_M':128,'BLOCK_N':64,'D':64},
        attrs={(i,): [['tt.divisibility',16]] for i in range(6)} if aligned else {})
    compiled=triton.compile(src,options={'num_warps':4,'num_stages':2})
    assert getattr(compiled.metadata,'global_scratch_size',0)==0
    launch=trtp.KernelLaunchParams()
    seq=q.shape_expr[2]
    launch.grid_x=(seq+127)//128
    launch.grid_y=q.shape_expr[0]*q.shape_expr[1]
    launch.block_x=compiled.metadata.num_warps*32
    launch.shared_mem=compiled.metadata.shared
    extras=trtp.SymIntExprs.from_tuple([trtp.SymInt32(seq)])
    return compiled.metadata.name.encode(),compiled.asm['ptx'].encode(),launch,extras


@trtp.register('nano_vla::vision_attention_aligned')
def describe_attention_aligned(q: trtp.TensorDesc, kt: trtp.TensorDesc, v: trtp.TensorDesc,
                               mask: trtp.TensorDesc) -> trtp.TensorDesc:
    return q.like()


@trtp.autotune('nano_vla::vision_attention_aligned')
def tune_attention_aligned(q: trtp.TensorDesc, kt: trtp.TensorDesc, v: trtp.TensorDesc,
                           mask: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]):
    return [trtp.AutoTuneCombination('FP16, FP16, FP16, FP16, FP16')]


@trtp.aot_impl('nano_vla::vision_attention_aligned')
def compile_attention_aligned(q: trtp.TensorDesc, kt: trtp.TensorDesc, v: trtp.TensorDesc,
                              mask: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int
                              ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    # Fixed 1024 sequence and contiguous HALF buffers; TensorRT's tested linear
    # tensor layout is 16-byte aligned. Preserve the generic variant for comparison.
    return _compile_attention(q, aligned=True)


@trtp.register('nano_vla::masked_row_softmax')
def describe_masked(inp: trtp.TensorDesc, mask: trtp.TensorDesc) -> trtp.TensorDesc:
    return inp.like()


@trtp.autotune('nano_vla::masked_row_softmax')
def tune_masked(inp: trtp.TensorDesc, mask: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]):
    return [trtp.AutoTuneCombination('FP16, FP16, FP16')]


@trtp.aot_impl('nano_vla::masked_row_softmax')
def compile_masked(inp: trtp.TensorDesc, mask: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc],
                   tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    import triton
    from aot_kernel import masked_softmax_aot
    src=triton.compiler.ASTSource(masked_softmax_aot,
        signature={'X':'*fp16','Mask':'*fp16','n_cols':'i32','row_stride':'i32','Y':'*fp16'},
        constexprs={'BLOCK_N':1024})
    compiled=triton.compile(src,options={'num_warps':4})
    cols=inp.shape_expr[inp.ndim-1]
    launch=trtp.KernelLaunchParams()
    launch.grid_x=inp.shape_expr.numel()//cols
    launch.block_x=compiled.metadata.num_warps*32
    launch.shared_mem=compiled.metadata.shared
    extras=trtp.SymIntExprs.from_tuple([trtp.SymInt32(cols),trtp.SymInt32(cols)])
    return compiled.metadata.name.encode(),compiled.asm['ptx'].encode(),launch,extras


# HALF 0.3535 (= 0.353515625), the exported Q and K scale; build_vision.py checks it.
VISION_SCALE = 0.353515625
NATIVE_TILE = dict(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3)  # swept on the board


@trtp.register('nano_vla::vision_attention_flat')
def describe_attention_flat(q: trtp.TensorDesc, k: trtp.TensorDesc,
                            v: trtp.TensorDesc) -> trtp.TensorDesc:
    return q.like()


@trtp.autotune('nano_vla::vision_attention_flat')
def tune_attention_flat(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc,
                        outputs: Tuple[trtp.TensorDesc]):
    return [trtp.AutoTuneCombination('FP16, FP16, FP16, FP16')]


@trtp.aot_impl('nano_vla::vision_attention_flat')
def compile_attention_flat(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc,
                           outputs: Tuple[trtp.TensorDesc], tactic: int
                           ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    # The same kernel on the [1,1024,768] projection outputs (identical memory layout
    # to the [1,1024,12,64] views), so no reshape copies are needed around it.
    return compile_attention_native(q, k, v, outputs, tactic)


@trtp.register('nano_vla::vision_attention_native')
def describe_attention_native(q: trtp.TensorDesc, k: trtp.TensorDesc,
                              v: trtp.TensorDesc) -> trtp.TensorDesc:
    return q.like()


@trtp.autotune('nano_vla::vision_attention_native')
def tune_attention_native(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc,
                          outputs: Tuple[trtp.TensorDesc]):
    return [trtp.AutoTuneCombination('FP16, FP16, FP16, FP16')]


@trtp.aot_impl('nano_vla::vision_attention_native')
def compile_attention_native(q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc,
                             outputs: Tuple[trtp.TensorDesc], tactic: int
                             ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    # Static [1,1024,12,64] HALF in the projection layout; TensorRT's linear buffers
    # are 16-byte aligned and every row stride (1536 B) is a multiple of 16.
    import triton
    from attention_kernel import attention_native_aot
    seq, heads, dim = 1024, 12, 64
    tile = NATIVE_TILE
    src=triton.compiler.ASTSource(attention_native_aot,
        signature={'Q':'*fp16','K':'*fp16','V':'*fp16','O':'*fp16'},
        constexprs={'N_CTX':seq,'H':heads,'SCALE':VISION_SCALE,'BLOCK_M':tile['BLOCK_M'],
                    'BLOCK_N':tile['BLOCK_N'],'D':dim},
        attrs={(i,): [['tt.divisibility',16]] for i in range(4)})
    compiled=triton.compile(src,options={'num_warps':tile['num_warps'],'num_stages':tile['num_stages']})
    assert getattr(compiled.metadata,'global_scratch_size',0)==0
    launch=trtp.KernelLaunchParams()
    launch.grid_x=seq//tile['BLOCK_M']
    launch.grid_y=heads
    launch.block_x=compiled.metadata.num_warps*32
    launch.shared_mem=compiled.metadata.shared
    return compiled.metadata.name.encode(),compiled.asm['ptx'].encode(),launch,trtp.SymIntExprs.from_tuple([])


@trtp.register('nano_vla::vision_attention_qkv')
def describe_attention_qkv(qkv: trtp.TensorDesc) -> trtp.TensorDesc:
    out = qkv.like()
    out.shape_expr[-1] = qkv.shape_expr[-1]//3
    return out


@trtp.autotune('nano_vla::vision_attention_qkv')
def tune_attention_qkv(qkv: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]):
    return [trtp.AutoTuneCombination('FP16, FP16')]


@trtp.aot_impl('nano_vla::vision_attention_qkv')
def compile_attention_qkv(qkv: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int
                          ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    # One [1,1024,2304] HALF projection (Q|K|V columns); every row stride is 16-byte aligned.
    import triton
    from attention_kernel import attention_qkv_aot
    seq, heads, dim = 1024, 12, 64
    tile = NATIVE_TILE
    src=triton.compiler.ASTSource(attention_qkv_aot,
        signature={'QKV':'*fp16','O':'*fp16'},
        constexprs={'N_CTX':seq,'H':heads,'SCALE':VISION_SCALE,'BLOCK_M':tile['BLOCK_M'],
                    'BLOCK_N':tile['BLOCK_N'],'D':dim},
        attrs={(i,): [['tt.divisibility',16]] for i in range(2)})
    compiled=triton.compile(src,options={'num_warps':tile['num_warps'],'num_stages':tile['num_stages']})
    assert getattr(compiled.metadata,'global_scratch_size',0)==0
    launch=trtp.KernelLaunchParams()
    launch.grid_x=seq//tile['BLOCK_M']
    launch.grid_y=heads
    launch.block_x=compiled.metadata.num_warps*32
    launch.shared_mem=compiled.metadata.shared
    return compiled.metadata.name.encode(),compiled.asm['ptx'].encode(),launch,trtp.SymIntExprs.from_tuple([])
