"""Run the existing Nsight trace entry point against an integrity-checked GR00T AOT cache."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from candidate import verify,validate_accumulate
from bench.vendor import groot_trt
from bench.tools.nsys_trace import main
args=sys.argv[1:]
cache=Path(args[args.index('--cache-dir')+1]).resolve()
bundle=Path(args[args.index('--bundle')+1]).resolve()
manifest=verify(cache,bundle)
validate_accumulate(manifest,args)
if '--accumulate' not in args:args += ['--accumulate',manifest['accumulate']]
groot_trt.ACCUMULATE=manifest['accumulate']
prebuild=groot_trt.prebuild_engines
def verified(b,c,*args,**kwargs):
    if Path(c).resolve()==cache:
        if b.root.resolve()!=bundle:raise ValueError('candidate cache belongs to another bundle')
        return {}
    return prebuild(b,c,*args,**kwargs)
groot_trt.prebuild_engines=verified
sys.argv[1:]=args
main()
