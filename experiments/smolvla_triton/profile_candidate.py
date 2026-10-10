"""Trace a verified experimental cache without rebuilding its patched engines."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.vendor import groot_trt
from bench.tools.nsys_trace import main
from candidate_cache import verify_candidate

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--cache-dir', type=Path, required=True)
args, _ = parser.parse_known_args()
cache = args.cache_dir.expanduser().resolve()
bundle = Path.home()/'bundles/smolvla-base-split'
manifest = verify_candidate(cache, bundle)
original = groot_trt.prebuild_engines

def verified_prebuild(b, path, *a, **kw):
    if Path(path).expanduser().resolve() == cache:
        if b.root.resolve() != bundle.resolve():
            raise ValueError('candidate cache belongs to a different bundle')
        return {}
    return original(b, path, *a, **kw)

groot_trt.prebuild_engines = verified_prebuild
main()
