"""Run a verified GR00T candidate cache in the torch-free TensorRT environment.

The wrapper verifies bundle inputs, engine bytes and timestep constants, then
skips native prebuild for that cache only. Native cache keys do not describe the
AOT graph replacements.
"""
from pathlib import Path
import importlib.util
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench import cli
from bench.vendor import groot_trt
from candidate import verify, validate_accumulate

args = sys.argv[1:]
out = Path(args[args.index('--out')+1]).expanduser()
if out.exists():
    raise FileExistsError(f'{out} already exists; use a unique output per run')
cache = Path(args[args.index('--cache-dir')+1]).expanduser().resolve()
bundle = Path(args[args.index('--bundle')+1]).expanduser().resolve()
manifest = verify(cache, bundle)
validate_accumulate(manifest, args)
if '--accumulate' not in args:
    args += ['--accumulate', manifest['accumulate']]
# Report what the cache was built with (older caches predate the option: auto).
groot_trt.ACCUMULATE = manifest.get('accumulate') or 'auto'
original = groot_trt.prebuild_engines


def verified(b, cache_dir, *a, **kw):
    if Path(cache_dir).expanduser().resolve() == cache:
        if b.root.resolve() != bundle:
            raise ValueError('candidate cache belongs to a different bundle')
        return {}
    return original(b, cache_dir, *a, **kw)


groot_trt.prebuild_engines = verified
from bench.backends.trt_split_groot import TrtSplitGrootBackend
meta = TrtSplitGrootBackend.meta
TrtSplitGrootBackend.meta = lambda self: {**meta(self), 'experimental_triton': manifest,
                                         'runtime_has_torch': importlib.util.find_spec('torch') is not None,
                                         'runtime_has_triton': importlib.util.find_spec('triton') is not None}
raise SystemExit(cli.main(args))
