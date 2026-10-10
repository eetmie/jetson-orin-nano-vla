"""Compare complete padded actions across images, prompts, noise and camera counts.

Uses verified experimental caches in the torch-free environment. Numerical stress
checks only: no actions are sent to hardware and synthetic actions have no task
quality meaning. Inputs are reproducible from the fixture, retained seed and code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.vendor import groot_trt
from bench.backends.trt_split_smolvla import TrtSplitSmolVLABackend
from candidate_cache import verify_candidate


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', action='append', required=True, help='LABEL=CACHE_PATH')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    bundle = Path.home()/'bundles/smolvla-base-split'
    caches = [('control', args.control.expanduser())]+[
        (arg.split('=', 1)[0], Path(arg.split('=', 1)[1]).expanduser()) for arg in args.candidate]
    manifests = {}
    for label, cache in caches:
        manifest = verify_candidate(cache, bundle)
        manifests[label] = manifest
    allowed = {cache.resolve() for _, cache in caches}
    original = groot_trt.prebuild_engines
    def verified_prebuild(b, cache, *a, **kw):
        if Path(cache).expanduser().resolve() in allowed:
            if b.root.resolve() != bundle.resolve():
                raise ValueError('candidate cache belongs to a different bundle')
            return {}
        return original(b, cache, *a, **kw)
    groot_trt.prebuild_engines = verified_prebuild
    backends = {}
    for label, cache in caches:
        be = TrtSplitSmolVLABackend(bundle, str(cache), chain='graph')
        be.load()
        backends[label] = be
    with np.load(bundle/'fixture.npz') as fixture:
        original_pixels = fixture['pixel_values'].astype(np.float32)
        original_state = fixture['model_state'].astype(np.float32)
        original_noise = fixture['noise'].astype(np.float32)
    rng = np.random.default_rng(20261010)
    checks = []
    for kind in ['fixture_pixels', 'black', 'grey', 'white', 'checkerboard', 'random_pixels']:
        pixels = original_pixels.copy()
        if kind in ['black', 'grey', 'white']:
            pixels.fill({'black':-1, 'grey':0, 'white':1}[kind])
        elif kind == 'checkerboard':
            yy, xx = np.indices(pixels.shape[-2:])
            pixels[:] = ((yy//16+xx//16)%2*2-1).astype(np.float32)
        elif kind == 'random_pixels':
            pixels[:] = rng.uniform(-1, 1, pixels.shape)
        for views in [1, 2]:
            task = 'pick up the blue block' if views == 1 else 'place the red block in the tray'
            state = original_state+rng.normal(0, 0.02, original_state.shape).astype(np.float32)
            noise = rng.standard_normal(original_noise.shape).astype(np.float32)
            images = [pixels[i:i+1] for i in range(views)]
            outputs = {}
            for label, be in backends.items():
                language = be.bundle.language(task)
                outputs[label] = be.device.infer(images, language, state, noise, key=task).copy()
            comparisons = []
            for label in backends:
                if label == 'control':
                    continue
                report = groot_trt._cmp(outputs[label], outputs['control'])
                assert np.isfinite(outputs[label]).all(), (kind, views, label)
                assert report['cosine'] >= .999 and report['max_pct_range'] <= 1, (kind, views, label, report)
                comparisons.append(dict(candidate=label, status='PASS', **report))
                print('PASS', kind, views, label, report, flush=True)
            checks.append(dict(image_kind=kind, views=views, task=task,
                pixels_sha256=digest(pixels), noise_sha256=digest(noise), state=state.tolist(),
                comparisons=comparisons, full_padded_actions={label:out.tolist() for label,out in outputs.items()}))
    report = dict(status='PASS', seed=20261010, scope='Full padded action comparison; numerical stress only',
                  fixture_sha256=hashlib.sha256((bundle/'fixture.npz').read_bytes()).hexdigest(),
                  manifests=manifests, checks=checks,
                  fixture_gates={label:be.fixture_parity for label,be in backends.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print('PASS ->', args.out)


if __name__ == '__main__':
    main()
