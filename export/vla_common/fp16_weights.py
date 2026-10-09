#!/usr/bin/env python3
# Vendored from the author's fine-tuning pipeline: vla-onnx/common/vla_common/fp16_weights.py @ 3f7793d.
"""Write mixed-FP16 copies of the split graphs, to halve what stays resident.

This halves the *deployed* footprint, not the build. The Orin measured that already:
TensorRT imports weights as FP32 working copies regardless of file dtype, so FP16
weights do not shrink the build peak — splitting is what makes the build fit. What
FP16 buys is residency, and on a 7.4 GB board with twelve sessions at ~6.7 GB plus a
camera and a control loop still to come, residency is the constraint.

**The precision recipe is a hard-won constant — do not "simplify" it.** A blanket FP16
cast overflowed SmolVLA's vision tower on this stack (cosine 0.805). Keeping
`LayerNormalization` and `Softmax` in FP32 mirrors the `trt_layer_norm_fp32_fallback`
the runtime already sets, and `keep_io_types=True` leaves graph inputs and outputs
FP32 so the runtime feeds nothing differently.

BF16 is not an option: on Orin (compute 8.7) `platform_has_fast_bf16` is n/a — there is
no hardware fast path — which is why the whole stack is FP16.

An FP16 weight cast is exactly the kind of change parity exists to catch. Re-run the
model's parity script against the converted bundle before trusting it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .bundle.manifest import write_manifest

# The two ops a blanket FP16 cast overflowed on this stack; kept FP32.
FP16_SENSITIVE_OPS = ("LayerNormalization", "Softmax")

_CHILD = r'''
import sys
from pathlib import Path
import onnx
from onnxruntime.transformers import float16
from onnxruntime.transformers.onnx_model import OnnxModel

src, dst, sensitive = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
model = onnx.load(str(src))
blocked = list(float16.DEFAULT_OP_BLOCK_LIST) + [s for s in sensitive if s]
converted = float16.convert_float_to_float16(
    model,
    keep_io_types=True,
    op_block_list=blocked,
)
dst.parent.mkdir(parents=True, exist_ok=True)
# save_model_to_file topologically sorts first, and that is load-bearing: keep_io_types
# appends its input Casts at the END of the node list, so a bare onnx.save writes a graph
# that consumes tensors before producing them. Inline whenever protobuf allows (< 2 GB),
# which is every split graph so far and the layout the deployed bundles have.
try:
    inline = converted.ByteSize() < 2_000_000_000
except Exception:
    inline = False
OnnxModel(converted).save_model_to_file(str(dst), use_external_data_format=not inline)
'''


def _onnx_mb(path: Path) -> float:
    """A graph's footprint, counting its external-data file if it has one."""
    ext = path.with_name(path.name + ".data")
    return (path.stat().st_size + (ext.stat().st_size if ext.is_file() else 0)) / 1e6


def convert_bundle(
    split_dir: Path,
    out_dir: Path,
    *,
    keep_fp32: tuple[str, ...] | list[str] = (),
    only: list[str] | None = None,
    copy_dirs: tuple[str, ...] = ("tokenizer", "processor"),
    sensitive_ops: tuple[str, ...] = FP16_SENSITIVE_OPS,
    verbose: bool = True,
) -> tuple[float, float]:
    """Convert every graph in `split_dir` to mixed FP16 under `out_dir`.

    `keep_fp32` names graphs copied unconverted (EVO1's `token_embedding` runs on the
    CPU EP and stays FP32). `only`, when given, converts just those and copies the
    rest. Returns (source MB, target MB).

    One subprocess per graph: the converter holds the FP32 model and its FP16 copy at
    once, and these run to several hundred MB each.
    """
    split_dir, out_dir = Path(split_dir).resolve(), Path(out_dir).resolve()
    bundle = json.loads((split_dir / "bundle.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in copy_dirs:
        src = split_dir / name
        if src.is_dir():
            dst = out_dir / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    fixture = (bundle.get("fixture") or {}).get("file")
    if fixture and (split_dir / fixture).exists():
        shutil.copy2(split_dir / fixture, out_dir / fixture)

    child = out_dir / "_convert_one.py"
    child.write_text(_CHILD)
    sensitive = ",".join(sensitive_ops)
    total_src = total_dst = 0.0
    try:
        for graph in bundle["graphs"]:
            src, dst = split_dir / graph["file"], out_dir / graph["file"]
            name = graph["name"]
            skip = name in keep_fp32 or (only is not None and name not in only)
            if skip:
                shutil.copy2(src, dst)
                mode = "copied (fp32)"
            else:
                proc = subprocess.run(
                    [sys.executable, str(child), str(src), str(dst), sensitive],
                    capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    tail = (proc.stderr or proc.stdout).strip().splitlines()[-8:]
                    raise SystemExit(
                        f"fp16 conversion failed for {name}:\n" + "\n".join(tail)
                    )
                mode = "mixed FP16"
            src_mb, dst_mb = _onnx_mb(src), _onnx_mb(dst)
            graph["size_mb"] = round(dst_mb, 1)
            total_src, total_dst = total_src + src_mb, total_dst + dst_mb
            if verbose:
                print(f"  {name:18s} {src_mb:6.0f} MB -> {dst_mb:6.0f} MB  {mode}")
    finally:
        child.unlink(missing_ok=True)

    (out_dir / "bundle.json").write_text(json.dumps(bundle, indent=2))
    n = write_manifest(out_dir)
    if verbose:
        ratio = f" ({total_dst / total_src:.2f}x)" if total_src else ""
        print(f"\nconverted {total_src:.0f} MB -> {total_dst:.0f} MB{ratio}")
        print(f"wrote {out_dir} ({n} files manifested)")
        print("\nNext: rebuild engines against this dir, then re-run parity.")
    return total_src, total_dst


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--keep-fp32", nargs="*", default=[],
                    help="graph names copied unconverted (e.g. token_embedding)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="convert only these graphs; the rest are copied as-is")
    args = ap.parse_args(argv)
    convert_bundle(args.split_dir, args.out_dir,
                   keep_fp32=tuple(args.keep_fp32), only=args.only)


if __name__ == "__main__":
    main()
