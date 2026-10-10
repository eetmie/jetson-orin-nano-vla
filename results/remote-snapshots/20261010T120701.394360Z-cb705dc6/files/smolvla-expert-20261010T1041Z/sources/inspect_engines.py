"""Retain TensorRT's actual layer descriptions and hashes for the tested caches."""
import argparse
import hashlib
import json
from pathlib import Path

import tensorrt as trt

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--cache', action='append', help='LABEL=PATH; repeat for each cache to inspect')
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
root = Path.home()/'.cache/jetson-orin-nano-vla'
baseline = root/'smolvla-base-trt'
entries = []
selected = [(value.split('=', 1)[0], Path(value.split('=', 1)[1]).expanduser()) for value in args.cache] if args.cache else [
    (mode, root/f'smolvla-triton-{mode}') for mode in ['control', 'fp16', 'masked']]
for mode, cache in selected:
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.glob('*.engine')}
    for name, digest in hashes.items():
        if name != 'vision.engine':
            assert digest == hashlib.sha256((baseline/name).read_bytes()).hexdigest(), name
    engine = runtime.deserialize_cuda_engine((cache/'vision.engine').read_bytes())
    assert engine is not None
    inspector = engine.create_engine_inspector()
    layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    entries.append(dict(mode=mode, cache=str(cache), engine_sha256=hashes, vision_layers=layers,
                        other_engines_identical_to_original=True))
    del inspector, engine
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(dict(tensorrt=trt.__version__, caches=entries), indent=2))
print('PASS ->', args.out)
