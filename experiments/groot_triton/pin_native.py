"""Pin a validated native GR00T cache for use as a matched stress-test control."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from candidate import pin, sha256, validate_base
from bench.vendor.groot_trt import _trt_version, _cudart_version

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', type=Path, required=True)
parser.add_argument('--cache-dir', type=Path, required=True)
args = parser.parse_args()
bundle, cache = args.bundle.expanduser().resolve(), args.cache_dir.expanduser().resolve()
output = cache / 'candidate.json'
if output.exists():
    raise FileExistsError(output)
stack = (f'tensorrt={_trt_version()} cudart={_cudart_version()} '
         'opt_level=2 workspace_mb=512 strongly_typed accumulate=fp32')
validate_base(cache, bundle, 'fp32', stack)
record = dict(mode='baseline', bundle=str(bundle), accumulate='fp32',
              tensorrt=_trt_version(), platform_stack=stack, graphs=[],
              precision='native TensorRT mixed FP16, FP32 accumulation',
              native_builder_sha256=sha256(Path(__file__).resolve().parents[2] / 'bench/vendor/groot_trt.py'),
              **pin(cache, bundle))
output.write_text(json.dumps(record, indent=2) + '\n')
print(f'Pinned native control: {cache}')
