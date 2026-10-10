"""Report overlapping memory views separately and compare matched saved actions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.parity import compare

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
root = args.results
data = {p.name:json.loads(p.read_text()) for p in sorted(root.glob('policy-*.json'))}
runs = []
for name,r in data.items():
    if r['status'] != 'ok':
        raise ValueError(f'failed run: {name}')
    system = r['system']['windows']['load']
    runs.append(dict(file=name, label=r['label'], duration_s=r['config']['duration_s'],
        metadata_collection='distribution metadata' if '-env-' in name else 'framework imports',
        used_for_selection='-env-' in name,
        latency_ms=r['latency_ms'], measurement=r['measurement'],
        board_ram_mb=system['ram_used_mb'],
        process_rss_mb=r['process']['windows']['load']['rss_mb'],
        stages_ms=r['latency_breakdown_ms'], telemetry=system,
        runtime={key:r['meta'].get(key) for key in ['lean_tokenizer', 'trim_host_heap',
            'host_heap_trim_released', 'prompt_cache_limit', 'runtime_imported_transformers',
            'runtime_has_torch', 'runtime_has_triton', 'shared_scratch_mb']},
        fixture_gate=r['meta']['fixture_parity']))
parity = []
for name,r in data.items():
    ref_name = 'policy-ffn-60s.json' if 'ffn' in name else 'policy-control-60s.json'
    if name == ref_name:
        continue
    ref = data[ref_name]
    report = compare(ref,r)
    if report['verdict'] != 'PASS':
        raise ValueError(report)
    report.update(file=name, reference_file=ref_name,
                  actions_bit_identical=np.array_equal(ref['saved_chunks']['chunks'], r['saved_chunks']['chunks']))
    if not report['actions_bit_identical']:
        raise ValueError('runtime memory changes must preserve exact actions')
    parity.append(report)
probes = []
for p in sorted(root.glob('probe-*.json')):
    r = json.loads(p.read_text())
    s = {row['label']:row for row in r['snapshots']}
    probes.append(dict(file=p.name, cache=r['cache'], lean_tokenizer=r.get('lean_tokenizer',False),
        runtime_imported_transformers=r.get('runtime_imported_transformers',True),
        actions_bit_identical=r['actions_bit_identical'],
        trim_rss_released_mib=(s['after_gc']['self/status']['VmRSS']-s['after_trim']['self/status']['VmRSS'])/2**20,
        snapshots=r['snapshots'], engines=r['engines']))
tokenizer = json.loads((root/'tokenizer-parity-bounded.json').read_text())
result = dict(scope='Two-camera SmolVLA, ten denoising steps; unchanged verified AOT engines with optional runtime memory changes',
    units='Board RAM and process RSS are reported by the existing monitor in MB; probe byte counts are converted explicitly to MiB.',
    memory_accounting='CPU RSS and CUDA allocations overlap on this unified-memory board: do not add them. Disk engine-cache size is not runtime RAM.',
    tokenizer={k:v for k,v in tokenizer.items() if k != 'checks'},
    runs=runs, saved_action_parity=parity, fresh_process_probes=probes,
    limitations=['Numerical checks do not measure robot task success.',
                  'Existing MAXN_SUPER clocks and board power conditions retained.',
                  'Board RAM includes other processes and varies between windows.',
                  'The initial policy-ffn-60s window may overlap a CPU-only tokenizer parity check; use corrected -env runs for RAM/speed selection.'])
args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps([dict(file=r['file'],p50_ms=r['latency_ms']['p50'],ram_mb=r['board_ram_mb']['mean'],rss_mb=r['process_rss_mb']['mean']) for r in runs],indent=2))
