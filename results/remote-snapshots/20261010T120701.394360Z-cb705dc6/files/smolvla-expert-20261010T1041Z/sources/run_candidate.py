"""Run an AOT candidate in the existing torch-free TRT environment.

Only this experiment bypasses ONNX prebuild after verifying the candidate's
source and engine hashes. The normal CLI rebuilds this cache from ONNX.
"""
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from bench import cli
from bench.vendor import groot_trt
from candidate_cache import verify_candidate

args=sys.argv[1:]
output=Path(args[args.index('--out')+1]).expanduser()
if output.exists():
    raise FileExistsError(f'{output} already exists; use a unique output per run')
cache=Path(args[args.index('--cache-dir')+1]).expanduser().resolve()
manifest=verify_candidate(cache)
source=Path.home()/'bundles/smolvla-base-split/smolvlm_vision.onnx'
# No plugin, Torch or Triton import: the serialized AOT engine must stand alone.
original=groot_trt.prebuild_engines

def verified_candidate(bundle,cache_dir,*args,**kwargs):
    if Path(cache_dir).expanduser().resolve()==cache:
        if bundle.root.resolve()!=source.parent.resolve():
            raise ValueError('candidate cache belongs to a different bundle')
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
