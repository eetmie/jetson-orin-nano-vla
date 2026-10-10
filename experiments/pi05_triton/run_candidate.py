"""Run a verified π0.5 candidate template cache in the torch-free TensorRT environment."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench import cli
from bench.vendor import pi05_trt

args = sys.argv[1:]
out = Path(args[args.index('--out')+1]).expanduser()
if out.exists():
    raise FileExistsError(f'{out} already exists; use a unique output per run')
cache = Path(args[args.index('--cache-dir')+1]).expanduser().resolve()
bundle = Path(args[args.index('--bundle')+1]).expanduser().resolve()
manifest = json.loads((cache/'candidate.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert manifest['bundle'] == str(bundle), 'candidate cache belongs to another bundle'
assert manifest['template_sha256'] == {p.name: sha(p) for p in sorted((bundle/'templates').glob('*.onnx'))}, 'templates changed'
assert manifest['engine_sha256'] == {p.name: sha(p) for p in sorted(cache.glob('*.engine')) if not p.name.startswith('op_')}, 'engines changed'
original = pi05_trt.prebuild_templates
pi05_trt.prebuild_templates = lambda b, c: {} if Path(c).expanduser().resolve() == cache else original(b, c)
from bench.backends.trt_split_pi05 import TrtSplitPi05Backend
meta = TrtSplitPi05Backend.meta
TrtSplitPi05Backend.meta = lambda self: {**meta(self), 'experimental_triton': manifest,
                                         'runtime_has_torch': importlib.util.find_spec('torch') is not None}
raise SystemExit(cli.main(args))
