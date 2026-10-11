"""Summarize paired GR00T speed/RAM measurements without rewriting raw runs."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
rows=[]
for tag in ['n16','n17']:
    runs={}
    for mode in ['baseline','all','vision','dit','cross']:
        for duration in ['screen','sustained']:
            path=a.directory/f'{tag}-{mode}-{duration}.json'
            if not path.exists():continue
            r=json.loads(path.read_text());assert r['status']=='ok',path
            report=r['meta']['fixture_parity']['reports']['chunk']
            row=dict(file=path.name,p50_ms=r['latency_ms']['p50'],p95_ms=r['latency_ms']['p95'],
                     hz=r['latency_ms']['hz_mean'],calls=r['latency_ms']['n'],
                     board_ram_mb=r['system']['windows']['load']['ram_used_mb'],
                     process_rss_mb=r['process']['windows']['load']['rss_mb'],
                     fixture_chunk=report,runtime_has_torch=r['meta'].get('runtime_has_torch'),
                     runtime_has_triton=r['meta'].get('runtime_has_triton'))
            runs[mode+'-'+duration]=row
    comparisons=[]
    for key,row in runs.items():
        mode,duration=key.split('-');control=runs.get('baseline-'+duration)
        if mode=='baseline' or not control:continue
        comparisons.append(dict(mode=mode,duration=duration,median_gain_pct=(1-row['p50_ms']/control['p50_ms'])*100,
            board_ram_delta_mb=row['board_ram_mb']['mean']-control['board_ram_mb']['mean'],
            process_rss_delta_mb=row['process_rss_mb']['mean']-control['process_rss_mb']['mean']))
    rows.append(dict(model=tag,runs=runs,comparisons=comparisons))
a.out.write_text(json.dumps(dict(models=rows),indent=2)+'\n')
print(json.dumps([dict(model=r['model'],comparisons=r['comparisons']) for r in rows],indent=2))
