"""Validate retained evidence and summarize the initial Nano Triton experiment."""
import hashlib
import json
from pathlib import Path

from bench.parity import load_results, parity_report

root = Path(__file__).resolve().parent
paths = [root/f'smolvla-base.triton-{mode}.json' for mode in ['control', 'fp16', 'masked']]
runs = [json.loads(path.read_text()) for path in paths]
assert all(r['status'] == 'ok' and r['meta']['fixture_parity']['status'] == 'PASS' for r in runs)
assert all(not r['meta']['runtime_has_torch'] and not r['meta']['runtime_has_triton'] for r in runs)
parity = parity_report(load_results(paths), runs[0]['label'])
assert all(c['verdict'] == 'PASS' and c['n_observations'] == 8 for c in parity['comparisons']), parity
inspection = json.loads((root/'engine-inspection.json').read_text())
operator_dir = root/'operator-rerun-20261010T0945Z'
operators = json.loads((operator_dir/'softmax-bench.json').read_text())
masked = json.loads((operator_dir/'masked-correctness.json').read_text())
assert operators['status'] == masked['status'] == 'PASS'
summary = dict(status='PASS', performance_verdict='Neither Triton policy candidate beats the TensorRT control',
               scope='Initial 60-second synthetic two-view ten-step graph runs on the actual SM87 Nano; no task-quality claim',
               parity=parity, policy_runs=[], standalone_softmax=[], masked_kernel_checks=masked,
               archival_note='All available remote measurements are also stored in unique local results/remote-snapshots directories. Earlier operator evidence remains under archives; the corrected same-input sweep is under operator-rerun-20261010T0945Z.')
for mode, run, inspected in zip(['control', 'fp16', 'masked'], runs, inspection['caches']):
    manifest = json.loads((root/f'candidate-{mode}.json').read_text())
    assert manifest['engine_sha256'] == inspected['engine_sha256']['vision.engine']
    assert inspected['other_engines_identical_to_original']
    if mode != 'control':
        source = root/f'sources-{mode}'
        for filename, key in [('aot_kernel.py', 'kernel_sha256'), ('plugin.py', 'plugin_sha256')]:
            assert hashlib.sha256((source/filename).read_bytes()).hexdigest() == manifest[key], (mode, key)
    load = run['system']['windows']['load']
    entry = dict(mode=mode, file=paths[['control','fp16','masked'].index(mode)].name,
                 latency_ms=run['latency_ms'], measurement=run['measurement'],
                 p50_change_pct=(run['latency_ms']['p50']/runs[0]['latency_ms']['p50']-1)*100,
                 fixture=run['meta']['fixture_parity'], runtime_has_torch=False, runtime_has_triton=False,
                 vision_layer_count=len(inspected['vision_layers']['Layers']), build_s=manifest['build_s'],
                 system_ram_mb=load['ram_used_mb'], process_rss_mb=run['process']['windows']['load']['rss_mb'],
                 temperature_c=load['temp_c'], throttle=load.get('throttle'))
    summary['policy_runs'].append(entry)
for row in operators['results']:
    best = min(row['triton'], key=lambda t:t['median_ms'])
    summary['standalone_softmax'].append(dict(dtype=row['dtype'], shape=row['shape'],
        trt_ms=row['baseline']['median_ms'], triton_ms=best['median_ms'], variant=best['variant'],
        warps=best['warps'], speedup=best['speedup_vs_trt'],
        caveat='Standalone softmax/casts only; production TensorRT fuses a wider sequence'))
(root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
for row in summary['policy_runs']:
    print(row['mode'], row['latency_ms']['p50'], 'ms', row['measurement']['achieved_hz'], 'Hz', row['p50_change_pct'], '% latency change')
print('PARITY', [c['verdict'] for c in parity['comparisons']])
