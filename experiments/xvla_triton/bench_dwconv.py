"""DaViT depthwise conv + residual in token layout vs torch conv2d (NCHW) on the four stage shapes."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernels import dwconv_tokens_aot
def bench(fn):
    fn(); torch.cuda.synchronize()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            for _ in range(50): fn()
    for _ in range(3): g.replay()
    ts = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b)/50)
    return sorted(ts)[3]
torch.manual_seed(8)
rows = []
for V, H, C in ((3, 56, 256), (3, 28, 512), (3, 14, 1024), (3, 7, 2048)):
    x = torch.randn(V, H*H, C, device='cuda').half(); w = (torch.randn(C, 1, 3, 3, device='cuda')*0.2).half(); b = (torch.randn(C, device='cuda')*0.1).half()
    nchw = x.transpose(1, 2).reshape(V, C, H, H)
    conv = torch.nn.functional.conv2d(nchw.float(), w.float(), b.float(), padding=1, groups=C).half()
    ref = (conv + nchw).reshape(V, C, H*H).transpose(1, 2)
    y = torch.empty_like(x)
    for bp, bc, nw in ((32, 64, 4), (64, 64, 4), (16, 128, 4), (32, 128, 4), (64, 32, 4), (16, 64, 2)):
        if C % bc: continue
        fn = lambda: dwconv_tokens_aot[((H*H+bp-1)//bp, C//bc, V)](x, w, b, y, N_PIX=H*H, WIDTH=H, C=C, BLOCK_P=bp, BLOCK_C=bc, num_warps=nw)
        t = bench(fn)
        rows.append(dict(shape=[V, H, H, C], bp=bp, bc=bc, w=nw, ms=t, max_abs=(y.float()-ref.float()).abs().max().item()))
        print(rows[-1], flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
