"""Torch-free AOT compilation for the Orin Nano's SM87 target."""


def compile_sm87(source, options):
    import triton
    from triton.backends.compiler import GPUTarget
    return triton.compile(source, target=GPUTarget("cuda", 87, 32), options=options)
