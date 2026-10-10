"""Summarize paired runs and graph/prompt checks; fail if evidence is incomplete."""
import json
from pathlib import Path

import numpy as np

from bench.parity import load_results, parity_report

root = Path(__file__).resolve().parent
models = ['smolvla-base', 'xvla-base', 'evo1-libero']
summary = {'paired_runs': [], 'shared_helper_smoke_tests': [], 'prompt_layout_checks': []}
for model in models:
    paths = [root/f'{model}.runtime-update.{v}.json' for v in ['baseline', 'candidate']]
    runs = [json.loads(p.read_text()) for p in paths]
    assert all(r['status'] == 'ok' and r['meta']['fixture_parity']['status'] == 'PASS' for r in runs)
    parity = parity_report(load_results(paths), runs[0]['label'])
    assert parity['comparisons'][0]['verdict'] == 'PASS', parity
    candidate = runs[1]
    load = candidate['system']['windows']['load']
    summary['paired_runs'].append({
        'model': model, 'baseline_p50_ms': runs[0]['latency_ms']['p50'],
        'candidate_p50_ms': candidate['latency_ms']['p50'],
        'candidate_p95_ms': candidate['latency_ms']['p95'],
        'hz': candidate['measurement']['achieved_hz'],
        'completed_calls': candidate['measurement']['completed_calls'],
        'window_s': candidate['measurement']['window_s'],
        'system_ram_mean_mb': load['ram_used_mb']['mean'],
        'process_rss_mb': candidate['process']['windows']['load']['rss_mb'],
        'temperatures_c': load['temp_c'], 'throttle': load.get('throttle'),
        'parity': parity['comparisons'][0],
        'bitwise_identical_saved_chunks': bool(np.array_equal(
            runs[0]['saved_chunks']['chunks'], candidate['saved_chunks']['chunks'])),
        'fixture': candidate['meta']['fixture_parity'],
    })
    check = json.loads((root/f'{model}.prompt-layouts.json').read_text())
    assert check['status'] == 'PASS', check
    memory = check['memory_samples']
    summary['prompt_layout_checks'].append({
        'model': model, 'cycles': check['cycles'], 'graphs': check['graphs'],
        'worst_cosine': min(c['cosine'] for c in check['checks']),
        'worst_max_pct_range': max(c['max_pct_range'] for c in check['checks']),
        'rss_first_kib': memory[0]['rss_kib'], 'rss_last_kib': memory[-1]['rss_kib'],
        'rss_min_kib': min(s['rss_kib'] for s in memory),
        'rss_max_kib': max(s['rss_kib'] for s in memory),
        'min_system_available_kib': min(s['available_kib'] for s in memory),
    })
for model in ['groot-n16-base', 'groot-n17-base', 'pi05-libero']:
    run = json.loads((root/f'{model}.runtime-update.candidate.json').read_text())
    assert run['status'] == 'ok' and run['meta']['fixture_parity']['status'] == 'PASS', model
    summary['shared_helper_smoke_tests'].append({
        'model': model, 'status': run['status'], 'fixture': run['meta']['fixture_parity'],
        'completed_calls': run['measurement']['completed_calls'],
        'p50_ms': run['latency_ms']['p50'],
    })
summary['manifest'] = json.loads((root/'validation-manifest.json').read_text())
summary['status'] = 'PASS'
(root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
for r in summary['paired_runs']:
    print(r['model'], 'p50', r['baseline_p50_ms'], '->', r['candidate_p50_ms'],
          'Hz', r['hz'], 'bitwise identical', r['bitwise_identical_saved_chunks'])
for r in summary['prompt_layout_checks']:
    print('PROBE', r)
for r in summary['shared_helper_smoke_tests']:
    print('SMOKE', r['model'], r['p50_ms'])
