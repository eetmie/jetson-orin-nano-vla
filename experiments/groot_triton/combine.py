"""Isolate vision and DiT effects by selecting immutable engines from two verified caches."""
import argparse,json,os,shutil
from pathlib import Path
from candidate import pin,verify
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--group',choices=['vision','dit'],required=True)
a=p.parse_args()
base,custom=[verify(p,a.bundle) for p in [a.control,a.candidate]]
assert base['accumulate']==custom['accumulate']
a.out.mkdir(parents=True,exist_ok=False)
selected=[]
for src in sorted(a.control.iterdir()):
    if not src.is_file() or src.name=='candidate.json':continue
    if src.suffix=='.engine':
        if src.stem.startswith(a.group):src=a.candidate/src.name;selected.append(src.stem)
        os.link(src,a.out/src.name)
    else:shutil.copy2(src,a.out/src.name)
manifest={**custom,'mode':a.group,'graphs':selected,'derived_from':[str(a.control.resolve()),str(a.candidate.resolve())],**pin(a.out,a.bundle)}
(a.out/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Created',a.group,a.out)
