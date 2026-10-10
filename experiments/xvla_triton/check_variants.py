"""Full action chunks of verified X-VLA caches on varied observations, against a control.

Numerical stress only (synthetic inputs): fixture pixels and prompt, black / grey /
white / checkerboard / random pixels, each with its own proprio and noise.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.vendor import groot_trt
from bench.backends.trt_split_xvla import TrtSplitXVLABackend
from candidate import verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, default=Path.home()/'bundles/xvla-base-split')
    p.add_argument('--control', type=Path, required=True, help='plain or candidate cache')
    p.add_argument('--candidate', action='append', required=True, help='LABEL=CACHE')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    bundle = a.bundle.expanduser().resolve()
    caches = [('control', a.control.expanduser().resolve())] + [
        (x.split('=', 1)[0], Path(x.split('=', 1)[1]).expanduser().resolve()) for x in a.candidate]
    manifests = {l: verify(c, bundle) for l, c in caches if (c/'candidate.json').exists()}
    original = groot_trt.prebuild_engines
    def verified(b, cache, *args, **kw):
        if Path(cache).expanduser().resolve() in manifests_caches:
            return {}
        return original(b, cache, *args, **kw)
    manifests_caches = {c for l, c in caches if l in manifests}
    groot_trt.prebuild_engines = verified
    backends = {}
    for label, cache in caches:
        be = TrtSplitXVLABackend(bundle, str(cache), chain='graph')
        be.load()
        backends[label] = be
    r = np.load(bundle/'fixture.npz')
    pix0, ids, prop0 = r['pixel_values'].astype(np.float32), r['input_ids'].astype(np.int64), r['proprio'].astype(np.float32)
    rng = np.random.default_rng(20261010)
    checks = []
    for kind in ['fixture', 'black', 'grey', 'white', 'checkerboard', 'random']:
        pix = pix0.copy()
        if kind in ('black', 'grey', 'white'):
            pix.fill({'black': pix0.min(), 'grey': 0.0, 'white': pix0.max()}[kind])
        elif kind == 'checkerboard':
            yy, xx = np.indices(pix.shape[-2:])
            pix[:] = np.where((yy//16+xx//16) % 2, pix0.max(), pix0.min())
        elif kind == 'random':
            pix[:] = rng.uniform(pix0.min(), pix0.max(), pix.shape)
        prop = prop0 + rng.normal(0, 0.02, prop0.shape).astype(np.float32)
        noise = rng.standard_normal(r['x1'].shape).astype(np.float32)
        outs = {l: be.device.infer(pix, ids, prop, noise).copy() for l, be in backends.items()}
        row = dict(kind=kind, comparisons={})
        for l, o in outs.items():
            if l == 'control':
                continue
            rep = groot_trt._cmp(o, outs['control'])
            rep['identical'] = bool(np.array_equal(o, outs['control']))
            assert np.isfinite(o).all() and rep['cosine'] >= .999 and rep['max_pct_range'] <= 1, (kind, l, rep)
            row['comparisons'][l] = rep
            print('PASS', kind, l, {k: rep[k] for k in ('cosine', 'max_pct_range', 'identical')}, flush=True)
        checks.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(status='PASS', manifests=manifests, checks=checks,
                                     fixture_gates={l: be.fixture_parity for l, be in backends.items()}), indent=2))
    print('PASS ->', a.out)


if __name__ == '__main__':
    main()
