"""Run an AOT candidate in the existing torch-free TRT environment.

Only this experiment bypasses ONNX prebuild after verifying the isolated vision
engine's manifest. Pointing the normal CLI at this cache will rebuild from ONNX.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from bench import cli
from bench.vendor import groot_trt

args=sys.argv[1:]
output=Path(args[args.index('--out')+1]).expanduser()
if output.exists():
    raise FileExistsError(f'{output} already exists; use a unique output per run')
cache=Path(args[args.index('--cache-dir')+1]).expanduser().resolve()
manifest=json.loads((cache/'candidate.json').read_text())
source=Path.home()/'bundles/smolvla-base-split/smolvlm_vision.onnx'
assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_onnx_sha256']
assert hashlib.sha256((cache/'vision.engine').read_bytes()).hexdigest()==manifest['engine_sha256']
# No plugin, Torch or Triton import: the serialized AOT engine must stand alone.
original=groot_trt.prebuild_engines

def verified_candidate(bundle,cache_dir,*args,**kwargs):
    if Path(cache_dir).expanduser().resolve()==cache:
        assert bundle.root.resolve()==source.parent.resolve()
        return {}
    return original(bundle,cache_dir,*args,**kwargs)

groot_trt.prebuild_engines=verified_candidate
from bench.backends.trt_split_smolvla import TrtSplitSmolVLABackend
meta=TrtSplitSmolVLABackend.meta

def experimental_meta(self):
    return {**meta(self),'experimental_triton':manifest,'runtime_has_torch':importlib.util.find_spec('torch') is not None,
            'runtime_has_triton':importlib.util.find_spec('triton') is not None}

TrtSplitSmolVLABackend.meta=experimental_meta
raise SystemExit(cli.main(args))
