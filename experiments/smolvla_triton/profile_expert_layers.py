"""Attribute the existing expert engine's time using TensorRT's layer profiler.

This changes synchronization and is diagnostic, not a policy throughput benchmark.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.vendor.smolvla_trt import Bundle, Policy
from bench.vendor.groot_trt import Engines
from candidate_cache import verify_candidate

parser = argparse.ArgumentParser()
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
bundle = Bundle(Path.home()/'bundles/smolvla-base-split')
verify_candidate(args.cache, bundle.root)
engines = Engines(bundle, args.cache.expanduser())
policy = Policy(bundle, engines.run)
fixture = np.load(bundle.root/'fixture.npz')
images = [policy.vision(x[None].astype(np.float32)) for x in fixture['pixel_values']]
language = (bundle.embed_ids(fixture['lang_tokens']), fixture['lang_masks'].astype(bool))
def sample():
    return policy.sample(images, language, fixture['model_state'].astype(np.float32), fixture['noise'])
sample()

class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.samples = defaultdict(list)
    def report_layer_time(self, name, ms):
        self.samples[name].append(ms)

profiler = Profiler()
engines.contexts['decode'].profiler = profiler
for _ in range(5):
    sample()
layers = sorted([dict(name=name, calls=len(values), mean_ms=float(np.mean(values)),
                     ms_per_infer=float(np.mean(values))*bundle.num_steps)
                 for name, values in profiler.samples.items()], key=lambda r:-r['ms_per_infer'])
report = dict(scope='TensorRT per-layer profiler, host execution; diagnostic only',
              cache=str(args.cache), num_steps=bundle.num_steps, layers=layers,
              total_decode_ms=sum(r['ms_per_infer'] for r in layers))
args.out.write_text(json.dumps(report, indent=2))
for row in layers[:25]:
    print(round(row['ms_per_infer'], 3), row['name'])
print('TOTAL', report['total_decode_ms'])
