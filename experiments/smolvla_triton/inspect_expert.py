"""Retain expert layer descriptions and verify the remaining engines are identical."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tensorrt as trt
from candidate_cache import sha256, verify_candidate

parser = argparse.ArgumentParser()
parser.add_argument('--base-cache', type=Path, required=True)
parser.add_argument('--cache', action='append', required=True, help='LABEL=CACHE_PATH')
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
base = args.base_cache.expanduser()
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
rows = []
for value in args.cache:
    label,path = value.split('=',1)
    cache = Path(path).expanduser()
    manifest = verify_candidate(cache)
    hashes = {p.name:sha256(p) for p in cache.glob('*.engine')}
    for name,digest in hashes.items():
        if name != 'decode.engine' and digest != sha256(base/name):
            raise ValueError(f'Non-expert engine changed: {name}')
    engine = runtime.deserialize_cuda_engine((cache/'decode.engine').read_bytes())
    assert engine is not None
    inspector = engine.create_engine_inspector()
    info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info['Layers']
    counts = dict(Counter(layer.get('LayerType') for layer in layers if isinstance(layer,dict)))
    print(label,'layers',len(layers),'types',counts,flush=True)
    rows.append(dict(label=label,cache=str(cache),engine_sha256=hashes,
        non_expert_engines_identical=True,layer_types=counts,layers=info,manifest=manifest))
    del inspector,engine
args.out.write_text(json.dumps(dict(tensorrt=trt.__version__,caches=rows),indent=2))
