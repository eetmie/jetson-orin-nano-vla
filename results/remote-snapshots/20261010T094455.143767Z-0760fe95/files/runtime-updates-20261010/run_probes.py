import json
import os
from pathlib import Path
import subprocess
import time
root=Path.home()/'vla-validation-20261010-small-runtime'
python=Path.home()/'jetson-orin-nano-vla/.venv-ort/bin/python'
manifest=root/'results/validation-manifest.json'
while True:
    data=json.loads(manifest.read_text())
    if any(run['returncode'] for run in data['runs']):
        raise SystemExit('Benchmarks failed; probes not started')
    if len(data['runs']) == 9:
        break
    time.sleep(5)
env=dict(os.environ, PYTHONPATH=str(root/'candidate'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OPENBLAS_NUM_THREADS='1')
for model in ['smolvla-base','xvla-base','evo1-libero']:
    print('START probes '+model,flush=True)
    cmd=[str(python),str(root/'check_prompt_layouts.py'),'--model',model,'--out',str(root/'results'/f'{model}.prompt-layouts.json'),'--cycles','40']
    with (root/'results'/f'{model}.prompt-layouts.log').open('w') as log:
        result=subprocess.run(cmd,cwd=root/'candidate',env=env,stdout=log,stderr=subprocess.STDOUT)
    text=(root/'results'/f'{model}.prompt-layouts.log').read_text()
    print(text[-4000:],flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
print('ALL PROMPT/LAYOUT PROBES PASS',flush=True)
