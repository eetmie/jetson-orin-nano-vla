"""Audit one verified candidate per fresh process, including freed host heap.

CUDA allocations and CPU RSS share physical RAM on the Nano; these overlapping
accounting views must not be added. MemAvailable includes reclaimable page cache.
No kernel/build environment is imported. No actions are sent to hardware.
"""
import argparse
import ctypes
import gc
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.vendor import groot_trt
from bench.backends.trt_split_smolvla import TrtSplitSmolVLABackend
from candidate_cache import verify_candidate


def snapshot(label):
    info = {}
    for name in ['meminfo', 'self/status', 'self/smaps_rollup']:
        fields = {}
        for line in Path('/proc', name).read_text().splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[-1] == 'kB':
                fields[parts[0].rstrip(':')] = int(parts[1])*1024
        info[name] = fields
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    groot_trt._ck(groot_trt._cuda().cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), 'cudaMemGetInfo')
    info.update(label=label, cuda_free_bytes=free.value, cuda_total_bytes=total.value,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    print(label, 'RSS MiB', round(info['self/status']['VmRSS']/2**20, 2),
          'available MiB', round(info['meminfo']['MemAvailable']/2**20, 2), flush=True)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lean-tokenizer', action='store_true')
    args = parser.parse_args()
    if args.lean_tokenizer:
        from runtime_memory import install_lean_tokenizer
        install_lean_tokenizer()
    if args.out.exists():
        raise FileExistsError(args.out)
    bundle = Path.home()/'bundles/smolvla-base-split'
    cache = args.cache_dir.expanduser().resolve()
    manifest = verify_candidate(cache, bundle)
    original = groot_trt.prebuild_engines
    def verified(b, path, *a, **kw):
        if Path(path).expanduser().resolve() == cache:
            if b.root.resolve() != bundle.resolve():
                raise ValueError('wrong candidate bundle')
            return {}
        return original(b, path, *a, **kw)
    groot_trt.prebuild_engines = verified
    rows = [snapshot('before_load')]
    be = TrtSplitSmolVLABackend(bundle, str(cache), chain='graph')
    be.load()
    rows.append(snapshot('after_load'))
    with np.load(bundle/'fixture.npz') as fixture:
        pixels = fixture['pixel_values'].astype(np.float32)
        state = fixture['model_state'].astype(np.float32)
        noise = fixture['noise'].astype(np.float32)
        lang = (be.bundle.embed_ids(fixture['lang_tokens']), fixture['lang_masks'].astype(bool))
    images = [pixels[i:i+1] for i in range(len(pixels))]
    def infer():
        start = time.perf_counter()
        actions = be.device.infer(images, lang, state, noise, key=('memory_probe',)).copy()
        return actions, (time.perf_counter()-start)*1000
    for _ in range(10):
        before, _ = infer()
    pre_times = [infer()[1] for _ in range(30)]
    rows.append(snapshot('after_warmup'))
    gc.collect()
    rows.append(snapshot('after_gc'))
    libc = ctypes.CDLL(None)
    trim = libc.malloc_trim
    trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
    trimmed = trim(0)
    rows.append(snapshot('after_trim'))
    after, _ = infer()
    if not np.array_equal(before, after):
        raise ValueError('host heap trim changed actions')
    post_times = [infer()[1] for _ in range(30)]
    rows.append(snapshot('after_post_trim_inference'))
    result = dict(cache=str(cache), manifest=manifest, snapshots=rows,
                  lean_tokenizer=args.lean_tokenizer,
                  runtime_imported_transformers='transformers' in sys.modules,
                  malloc_trim_return=trimmed, actions_bit_identical=True,
                  fixture_gate=be.fixture_parity, trim_scope='once after initialization',
                  timings_ms=dict(before=pre_times, after=post_times),
                  shared_scratch_bytes=be.engines.scratch_bytes,
                  engines={name:dict(scratch_bytes=e.device_memory_size_v2,
                                    streamable_weights_bytes=e.streamable_weights_size,
                                    serialized_bytes=(cache/(name+'.engine')).stat().st_size
                                    if (cache/(name+'.engine')).exists() else None)
                           for name,e in be.engines.engines.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
