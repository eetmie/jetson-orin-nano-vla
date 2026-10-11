"""Compare complete GR00T actions with deterministic images/state/noise and frame timestamps."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from candidate import verify
from bench.backends.trt_split_groot import TrtSplitGrootBackend
from bench.vendor import groot_trt
from bench.vendor.imaging import map_views
from bench.obs import make_obs

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path)
a=p.parse_args()
if a.out.exists() or a.out.with_suffix('.npz').exists():raise FileExistsError(a.out)
cache,bundle=a.cache_dir.resolve(),a.bundle.resolve()
manifest=verify(cache,bundle)
groot_trt.ACCUMULATE=manifest['accumulate']
prebuild=groot_trt.prebuild_engines
def verified(b,c,*args,**kwargs):
    if Path(c).resolve()==cache:
        assert b.root.resolve()==bundle;return {}
    return prebuild(b,c,*args,**kwargs)
groot_trt.prebuild_engines=verified
be=TrtSplitGrootBackend(bundle,str(cache),chain='graph');be.load()
b=be.bundle.b
obs=make_obs('synthetic',b['task'],b['action_horizon'],b['state_dim'],max_action_dim=b['action_dim'],seed=20261011,n_views=3)
chunks=[];cases=[]
for i in range(12):
    o=obs[i]
    images=o.images
    variant=i%4
    if variant==1:images=[255-x for x in images]
    elif variant==2:images=[np.zeros_like(x) if j==0 else x[:,::-1].copy() for j,x in enumerate(images)]
    elif variant==3:images=[np.full_like(x,255) if j==2 else np.roll(x,23,axis=0) for j,x in enumerate(images)]
    pv=np.stack(map_views(be.bundle.preprocess,images))
    state=be._state(o)
    state*= [0.,.2,1.,-1.][variant]
    noise=o.noise*[0.,.2,1.,1.5][variant]
    timestamp=100.+i*.23
    if be.n17:chunk,age=be.device.infer(pv,state,noise,timestamp)
    else:chunk=be.device.infer(pv,state,noise);age=None
    assert np.isfinite(chunk).all(),f'case {i}: nonfinite actions'
    chunks.append(chunk.copy())
    cases.append(dict(index=i,image_variant=variant,state_scale=[0.,.2,1.,-1.][variant],noise_scale=[0.,.2,1.,1.5][variant],capture_s=timestamp,history_age_s=age))
    print('case',i,'pass',flush=True)
report=dict(status='pass',seed=20261011,thresholds=dict(cosine_min=.99999,max_action_range_pct=.27),model=b['model'],fixture_parity=be.fixture_parity,cases=cases,cache_manifest=manifest,shape=list(np.asarray(chunks).shape))
if a.reference:
    ref=np.load(a.reference.with_suffix('.npz'))['chunks'];assert ref.shape==np.asarray(chunks).shape
    ref_report=json.loads(a.reference.read_text());assert ref_report['cases']==cases,'history/input cases differ'
    reports=[];failures=[]
    for i,(x,y) in enumerate(zip(chunks,ref)):
        r=groot_trt._cmp(x,y);r.update(index=i)
        if be.n17:r['used_dims']=groot_trt._cmp(x[...,:b['action_used']],y[...,:b['action_used']])
        if r['cosine']<.99999 or r['max_pct_range']>.27:failures.append(i)
        if be.n17 and (r['used_dims']['cosine']<.99999 or r['used_dims']['max_pct_range']>.27):failures.append(i)
        reports.append(r)
    report['vs_control']=reports
    report['failures']=sorted(set(failures));report['status']='fail' if failures else 'pass'
np.savez(a.out.with_suffix('.npz'),chunks=np.asarray(chunks))
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(report['status'].upper(),a.out,flush=True)
raise SystemExit(0 if report['status']=='pass' else 1)
