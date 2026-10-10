"""On-board control-flow regressions using cached engines and TRT host references.

Run from the candidate checkout with its bench package on PYTHONPATH. This script
uses synthetic/fixture tensors; it does not send actions to a robot.
"""
import argparse
import ctypes
import json
from pathlib import Path
import time

import numpy as np

from bench.cli import Resolved, trt_backend
from bench.vendor.groot_trt import _cmp


class CudaGraphs:
    """Count actual runtime graph handles while forwarding every call to CUDA."""
    def __init__(self, cu, current):
        self.cu = cu
        self.sources = set()
        self.executables = {current.value} if current is not None else set()
        self.created = self.destroyed = self.sources_destroyed = 0
        self.peak_executables = len(self.executables)

    def __getattr__(self, name):
        return getattr(self.cu, name)

    def cudaStreamEndCapture(self, stream, output):
        rc = self.cu.cudaStreamEndCapture(stream, output)
        if rc == 0:
            handle = ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0]
            if handle:
                self.sources.add(handle)
        return rc

    def cudaGraphInstantiate(self, output, graph, flags):
        rc = self.cu.cudaGraphInstantiate(output, graph, flags)
        if rc == 0:
            self.executables.add(ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0])
            self.created += 1
            self.peak_executables = max(self.peak_executables, len(self.executables))
        return rc

    def cudaGraphDestroy(self, graph):
        rc = self.cu.cudaGraphDestroy(graph)
        if rc == 0:
            self.sources.remove(graph.value)
            self.sources_destroyed += 1
        return rc

    def cudaGraphExecDestroy(self, graph):
        rc = self.cu.cudaGraphExecDestroy(graph)
        if rc == 0:
            self.executables.remove(graph.value)
            self.destroyed += 1
        return rc

    def report(self):
        return dict(created=self.created, retired=self.destroyed,
                    source_graphs_destroyed=self.sources_destroyed,
                    live_source_graphs=len(self.sources),
                    live_executables=len(self.executables),
                    peak_executables=self.peak_executables)


def memory():
    def field(path, name):
        for line in Path(path).read_text().splitlines():
            if line.startswith(name + ':'):
                return int(line.split()[1])
    return dict(rss_kib=field('/proc/self/status', 'VmRSS'),
                available_kib=field('/proc/meminfo', 'MemAvailable'))


p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--cycles', type=int, default=40)
a = p.parse_args()
main = Path.home() / 'jetson-orin-nano-vla'
config = json.loads((main/'results'/f'{a.model}.trt.json').read_text())
a.task, a.views = config['model']['task'], config['model']['views']
a.cache_dir, a.chain = config['meta']['engine_cache'], 'graph'
bundle_path = Path(config['meta']['bundle'])
backend = trt_backend(a, Resolved(a, bundle_path), bundle_path)
backend.load()
b, dev = backend.bundle, backend.device
manifest = b.info if a.model == 'smolvla-base' else b.b
with np.load(bundle_path/manifest['fixture']['file']) as f:
    fixture = {k: f[k].copy() for k in f.files}
graphs = CudaGraphs(dev.d.cu, dev.graph)
dev.d.cu = graphs
checks = []


def check(name, actual, expected):
    report = _cmp(actual, expected)
    assert np.isfinite(actual).all(), name
    assert report['cosine'] >= .999 and report['max_pct_range'] <= 1., (name, report)
    checks.append(dict(name=name, **report))


if a.model == 'smolvla-base':
    pix = fixture['pixel_values'].astype(np.float32)
    pixels = [pix[i:i+1] for i in range(len(pix))]
    state, noise = fixture['model_state'].astype(np.float32), fixture['noise']
    lang_a = (b.embed_ids(fixture['lang_tokens']), fixture['lang_masks'].astype(bool))
    lang_b = b.language('pick up the blue block')
    embs = [backend.policy.vision(x) for x in pixels]
    cases = [(pixels, lang_a), (pixels[:1], lang_b)]
    refs = [backend.policy.sample(embs[:len(ps)], lang, state, noise) for ps, lang in cases]
    fixed_camera_ref = backend.policy.sample(embs, lang_b, state, noise)
    check('prompt without key: fixture', dev.infer(pixels, lang_a, state, noise), refs[0])
    check('prompt without key: changed language',
          dev.infer(pixels, lang_b, state, noise), fixed_camera_ref)
    for i, ((ps, lang), ref) in enumerate(zip(cases, refs)):
        check(f'camera layout warmup {i}', dev.infer(ps, lang, state, noise, key=f'layout-{i}'), ref)
    def cycle(i):
        ps, lang = cases[i % 2]
        return dev.infer(ps, lang, state, noise, key=f'layout-{i % 2}'), refs[i % 2]
elif a.model == 'evo1-libero':
    from bench.vendor.evo1_trt import infer, causal_mask
    pix, state = fixture['pixel_values'].astype(np.float32), fixture['state'].astype(np.float32)
    noise = fixture['initial_noise']
    ids, mask = fixture['input_ids'].astype(np.int64), fixture['context_mask'].astype(bool)
    modified_mask = mask.copy()
    text = np.flatnonzero(mask[0] & (ids[0] != b.image_token))
    assert len(text) > 0
    modified_mask[0, text[-1]] = False
    cases = [(ids, mask), (ids, modified_mask), (np.roll(ids, 1, axis=1), np.roll(mask, 1, axis=1))]
    refs = [infer(b, backend.engines.run, pix, ti, tm, state, noise)['action'] for ti, tm in cases]
    for i, ((ti, tm), ref) in enumerate(zip(cases, refs)):
        check(f'prompt/mask/layout warmup {i}', dev.infer(pix, ti, tm, state, noise), ref)
        dev.d.download(dev.cmask)
        dev.d.download(dev.mask)
        dev.d.sync()
        np.testing.assert_array_equal(dev.cmask.host(), tm)
        np.testing.assert_array_equal(dev.mask.host(), causal_mask(tm))
    def cycle(i):
        # Alternate between two image-copy layouts to force actual recapture.
        index = 0 if i % 2 == 0 else 2
        ti, tm = cases[index]
        return dev.infer(pix, ti, tm, state, noise), refs[index]
elif a.model == 'xvla-base':
    from bench.vendor.xvla_trt import infer
    pix, state = fixture['pixel_values'].astype(np.float32), fixture['proprio'].astype(np.float32)
    noise = fixture['x1']
    ids_a = fixture['input_ids'].astype(np.int64)
    ids_b = b.input_ids('pick up the blue block')
    assert not np.array_equal(ids_a, ids_b)
    refs = [infer(b, backend.engines.run, pix, ti, state, noise) for ti in [ids_a, ids_b]]
    uploads = [0]
    upload = dev.d.upload
    def count_upload(dst, arr):
        if dst is dev.ids:
            uploads[0] += 1
        return upload(dst, arr)
    dev.d.upload = count_upload
    mutable_ids = ids_a.copy()
    check('initial prompt', dev.infer(pix, mutable_ids, state, noise), refs[0])
    before = uploads[0]
    check('repeated prompt', dev.infer(pix, mutable_ids, state, noise), refs[0])
    assert uploads[0] == before, 'unchanged IDs were uploaded'
    mutable_ids[...] = ids_b
    check('in-place changed prompt', dev.infer(pix, mutable_ids, state, noise), refs[1])
    assert uploads[0] == before + 1, 'changed IDs were not uploaded'
    def cycle(i):
        index = (i // 2) % 2  # repeated IDs as well as prompt switches
        mutable_ids[...] = [ids_a, ids_b][index]
        return dev.infer(pix, mutable_ids, state, noise), refs[index]
else:
    raise ValueError(a.model)

samples = [memory()]
start = time.perf_counter()
for i in range(a.cycles):
    actual, expected = cycle(i)
    check(f'stress {i}', actual, expected)
    samples.append(memory())
    assert not graphs.sources, graphs.report()
    assert len(graphs.executables) == 1, graphs.report()
def fail_enqueue():
    raise ValueError('intentional enqueue failure')
try:
    dev.d.capture(fail_enqueue)
except ValueError as exc:
    assert str(exc) == 'intentional enqueue failure'
else:
    raise AssertionError('capture swallowed the enqueue failure')
assert not dev.d.capturing and not graphs.sources, graphs.report()
assert len(graphs.executables) == 1, graphs.report()
result = dict(status='PASS', model=a.model, cycles=a.cycles,
              capture_exception_cleanup=True,
              elapsed_s=time.perf_counter()-start,
              fixture=backend.fixture_parity, checks=checks, graphs=graphs.report(),
              memory_samples=samples, observations='bundle fixture; synthetic prompt/layout perturbations',
              reference='same Nano TensorRT engines, host chain')
a.out.write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps({k: result[k] for k in ['status', 'model', 'cycles', 'elapsed_s', 'graphs']}, indent=2))
