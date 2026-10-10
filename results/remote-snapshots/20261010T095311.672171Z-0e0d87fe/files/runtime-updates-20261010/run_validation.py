import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path.home() / 'vla-validation-20261010-small-runtime'
main = Path.home() / 'jetson-orin-nano-vla'
python = main / '.venv-ort/bin/python'
out = root / 'results'
env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OPENBLAS_NUM_THREADS='1')
models = ['smolvla-base', 'xvla-base', 'evo1-libero', 'groot-n16-base', 'groot-n17-base', 'pi05-libero']
metadata = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'base_commit': subprocess.check_output(['git', '-C', str(main), 'rev-parse', 'HEAD'], text=True).strip(),
            'candidate_runtime_sha256': hashlib.sha256((root/'candidate/bench/vendor/trt_device.py').read_bytes()).hexdigest(),
            'baseline_runtime_sha256': hashlib.sha256((root/'baseline/bench/vendor/trt_device.py').read_bytes()).hexdigest(),
            'runs': []}
(out/'validation-manifest.json').write_text(json.dumps(metadata, indent=2))
for model in models:
    config = json.loads((main/'results'/f'{model}.trt.json').read_text())
    variants = ['baseline', 'candidate'] if model in models[:3] else ['candidate']
    for variant in variants:
        label = f'{model}.runtime-update.{variant}'
        cmd = [str(python), '-m', 'bench', 'trt-split', '--model', model,
               '--bundle', config['meta']['bundle'], '--cache-dir', config['meta']['engine_cache'],
               '--chain', 'graph', '--views', str(config['model']['views']),
               '--task', config['model']['task'], '--label', label,
               '--warmup', '10' if model in models[:3] else '3', '--idle-s', '3',
               '--out', str(out/f'{label}.json')]
        cmd += ['--duration-s', '60'] if model in models[:3] else ['--iters', '5']
        print(f'START {label}', flush=True)
        started = time.time()
        with (out/f'{label}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=root/variant, env=env, stdout=log, stderr=subprocess.STDOUT)
        metadata['runs'].append({'label': label, 'command': cmd, 'returncode': result.returncode,
                                 'elapsed_s': time.time()-started})
        (out/'validation-manifest.json').write_text(json.dumps(metadata, indent=2))
        if result.returncode:
            print((out/f'{label}.log').read_text()[-6000:], flush=True)
            raise SystemExit(result.returncode)
        run = json.loads((out/f'{label}.json').read_text())
        print(f"DONE {label}: status={run['status']} p50={run['latency_ms']['p50']} p95={run['latency_ms']['p95']} n={run['latency_ms'].get('n')} fixture={run.get('meta',{}).get('fixture_parity',{}).get('status')}", flush=True)
print('ALL BENCHMARKS PASS', flush=True)
