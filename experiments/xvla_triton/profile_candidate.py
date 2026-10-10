"""Nsight trace of a verified X-VLA candidate cache (same args as bench.tools.nsys_trace)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.vendor import groot_trt
from bench.tools.nsys_trace import main
from candidate import verify

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--cache-dir', type=Path, required=True)
parser.add_argument('--bundle', type=Path, required=True)
args, _ = parser.parse_known_args()
cache, bundle = args.cache_dir.expanduser().resolve(), args.bundle.expanduser().resolve()
verify(cache, bundle)
original = groot_trt.prebuild_engines
groot_trt.prebuild_engines = lambda b, path, *a, **kw: {} if Path(path).expanduser().resolve() == cache \
    else original(b, path, *a, **kw)
main()
