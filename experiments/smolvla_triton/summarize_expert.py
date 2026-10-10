"""Summarize retained expert experiments without changing their raw results."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.parity import load_results, parity_report

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
root = args.results
runs = []
for p in sorted(root.glob('policy-*-*s.json')):
    r = json.loads(p.read_text())
    runs.append(dict(file=p.name,label=r['label'],status=r['status'],
        latency_ms=r.get('latency_ms'),stages_ms=r.get('latency_breakdown_ms'),
        measurement=r.get('measurement'),fixture_parity=r.get('meta',{}).get('fixture_parity'),
        runtime_has_torch=r.get('meta',{}).get('runtime_has_torch'),
        runtime_has_triton=r.get('meta',{}).get('runtime_has_triton'),
        process=r.get('process'),system=r.get('system')))
variants = json.loads((root/'policy-variants-01.json').read_text())
operator = json.loads((root/'ffn-operator-packed-02.json').read_text())
valid = [r for r in operator['candidates'] if 'median_ms' in r]
best = min(valid,key=lambda r:r['median_ms'])
parity = []
for duration in [60,300]:
    paths = sorted(root.glob(f'policy-*-{duration}s.json'))
    if len(paths) < 2:
        continue
    reference = next((p for p in paths if p.name==f'policy-expert-control-{duration}s.json'), paths[0])
    parity.append(dict(duration_s=duration,**parity_report(load_results(paths),json.loads(reference.read_text())['label'])))
comparisons = []
for duration in [60,300]:
    by_file = {r['file']:r for r in runs}
    ref = by_file.get(f'policy-expert-control-{duration}s.json')
    candidate = by_file.get(f'policy-expert-ffn-{duration}s.json')
    if ref and candidate:
        comparisons.append(dict(duration_s=duration,
            control_p50_ms=ref['latency_ms']['p50'],candidate_p50_ms=candidate['latency_ms']['p50'],
            latency_reduction_pct=(1-candidate['latency_ms']['p50']/ref['latency_ms']['p50'])*100))
summary = dict(scope='Two-camera SmolVLA, ten denoising steps, existing aligned Triton vision; expert projection experiment',
    variant_checks=dict(status=variants['status'],cases=len(variants['checks']),
        max_error_pct_of_range=max(c['max_pct_range'] for row in variants['checks'] for c in row['comparisons']),
        cosine_min=min(c['cosine'] for row in variants['checks'] for c in row['comparisons'])),
    isolated_projection=dict(scope=operator['scope'],control=operator['baseline'],best=best),
    runs=runs,paired_comparisons=comparisons,saved_action_parity=parity)
extended = root/'ffn-operator-extended-03.json'
if extended.exists():
    sweep = json.loads(extended.read_text())
    successful = [row for row in sweep['candidates'] if 'median_ms' in row]
    summary['extended_projection_sweep'] = dict(control_ms=sweep['baseline']['median_ms'],
        best=min(successful,key=lambda row:row['median_ms']))
args.out.write_text(json.dumps(summary,indent=2))
print(json.dumps({'variants':summary['variant_checks'],'paired_comparisons':comparisons,
                  'saved_action_verdicts':[c['verdict'] for r in parity for c in r['comparisons']]},indent=2))
