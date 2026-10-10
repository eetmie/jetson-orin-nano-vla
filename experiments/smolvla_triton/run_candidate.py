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
lean_tokenizer='--lean-tokenizer' in args
trim_heap='--trim-host-heap' in args
for flag in ['--lean-tokenizer', '--trim-host-heap']:
    if flag in args:
        args.remove(flag)
from runtime_memory import PROMPT_CACHE_LIMIT, install_lean_tokenizer, trim_host_heap
if lean_tokenizer:
    install_lean_tokenizer()
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
if trim_heap:
    original_load=TrtSplitSmolVLABackend.load
    def load_and_trim(self):
        original_load(self)
        self.host_heap_trim_released=trim_host_heap()
    TrtSplitSmolVLABackend.load=load_and_trim

def experimental_meta(self):
    return {**meta(self),'experimental_triton':manifest,'runtime_has_torch':importlib.util.find_spec('torch') is not None,
            'runtime_has_triton':importlib.util.find_spec('triton') is not None,
            'lean_tokenizer':lean_tokenizer, 'trim_host_heap':trim_heap,
            'prompt_cache_limit':PROMPT_CACHE_LIMIT if lean_tokenizer else None,
            'runtime_imported_transformers':'transformers' in sys.modules,
            'host_heap_trim_released':getattr(self,'host_heap_trim_released',None)}

TrtSplitSmolVLABackend.meta=experimental_meta
raise SystemExit(cli.main(args))
