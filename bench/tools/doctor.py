"""Check real imports and GPU access for the native VLA benchmark workflow.

Run separately from inference: dependency probes should not affect measured RAM.
Torch stays out of the native environments. Tokenizer families also probe Transformers.
JetPack libraries remain system dependencies.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import platform
import sys
from pathlib import Path


def probe(role: str, model: str = "smolvla-base") -> dict:
    from bench.models import get
    family = get(model).family
    checks = []

    def check(name, fn):
        try:
            value = fn()
            checks.append(dict(name=name, status="pass", detail=str(value)))
        except Exception as exc:
            checks.append(dict(name=name, status="fail", detail=f"{type(exc).__name__}: {exc}"))

    def module(name):
        mod = importlib.import_module(name)
        return f"{getattr(mod, '__version__', 'imported')} from {mod.__file__}"

    for name in ("numpy", "cv2", "tokenizers", "psutil", "tensorrt"):
        check(name, lambda name=name: module(name))
    if family in ("xvla", "evo1"):
        for name in ("transformers", "PIL", "sentencepiece"):
            check(name, lambda name=name: module(name))
    check("CLI and native backend imports", lambda: (
        module("bench.cli"), module(f"bench.backends.trt_split_{family}"),
        module(f"bench.vendor.{family}_trt"), module("bench.vendor.trt_device")))

    def cuda():
        driver = ctypes.CDLL("libcuda.so.1")
        def call(name, *args):
            rc = getattr(driver, name)(*args)
            if rc:
                raise RuntimeError(f"{name}: CUDA error {rc}")
        call("cuInit", 0)
        dev, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        call("cuDeviceGet", ctypes.byref(dev), 0)
        call("cuDeviceGetAttribute", ctypes.byref(major), 75, dev)
        call("cuDeviceGetAttribute", ctypes.byref(minor), 76, dev)
        if (major.value, minor.value) != (8, 7):
            raise RuntimeError(f"this recipe targets SM87; device is SM{major.value}{minor.value}")
        from bench.vendor.groot_trt import _cuda, _ck
        ptr = ctypes.c_void_p()
        _ck(_cuda().cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(4096)), "doctor allocation")
        _ck(_cuda().cudaFree(ptr), "doctor free")
        return "SM87, CUDA 13 runtime allocation/free passed"

    check("CUDA driver and runtime", cuda)
    if role == "build" and family != "groot":
        check("triton", lambda: module("triton"))

        def aot():
            import tensorrt.plugin as trtp
            # The actual selected plugin registrations expose transitive imports.
            root = Path(__file__).resolve().parents[2] / f"experiments/{family}_triton"
            sys.path.insert(0, str(root))
            importlib.import_module("plugin")
            if family == "smolvla":
                for name in ("ffn_plugin", "attention_kernel", "ffn_kernel"):
                    importlib.import_module(name)
            else:
                importlib.import_module("kernels")
            return f"AOT API available: {callable(trtp.aot_impl)}"
        check("TensorRT AOT and selected kernels", aot)
        def compile_probe():
            import triton
            root = Path(__file__).resolve().parents[2] / "experiments/smolvla_triton"
            sys.path.insert(0, str(root))
            from aot_kernel import softmax_aot
            from bench.vendor.triton_compile import compile_sm87
            source = triton.compiler.ASTSource(softmax_aot,
                signature={"X": "*fp16", "n_cols": "i32", "row_stride": "i32", "Y": "*fp16"},
                constexprs={"BLOCK_N": 1024})
            compiled = compile_sm87(source, {"num_warps": 4})
            if not compiled.asm.get("ptx"):
                raise RuntimeError("compiler produced no PTX")
            return "SM87 PTX generated without automatic driver discovery"
        check("AOT compilation", compile_probe)
    return dict(role=role, model=model, python=sys.version.split()[0], executable=sys.executable,
                platform=platform.platform(), checks=checks,
                status="pass" if all(c["status"] == "pass" for c in checks) else "fail")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("runtime", "build"), default="runtime")
    from bench.models import REGISTRY
    parser.add_argument("--model", choices=sorted(REGISTRY), default="smolvla-base")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = probe(args.role, args.model)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        with args.out.open("x") as stream:
            stream.write(rendered + "\n")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
