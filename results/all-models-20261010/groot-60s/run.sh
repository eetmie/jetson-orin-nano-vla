#!/bin/bash
cd ~/jetson-orin-nano-vla
C=/home/joel/.cache/jetson-orin-nano-vla
R=/home/joel/audit-20261010/groot
run() {  # label model bundle cache
  .venv-ort/bin/python -m bench trt-split --model $2 --bundle /home/joel/bundles/$3 --cache-dir $C/$4 --chain graph --warmup 10 --idle-s 3 --duration-s 60 --label $1 --out $R/$1.json > $R/$1.log 2>&1
  echo "$1 rc=$?"; grep "built" $R/$1.log | tr "\n" " "; echo; tail -1 $R/$1.log
  python3 -c "
import json;d=json.load(open(\"$R/$1.json\"));fp=d[\"meta\"][\"fixture_parity\"];dc=fp.get(\"device_chain\",{})
print(d[\"latency_breakdown_ms\"], d[\"process\"][\"windows\"][\"load\"][\"rss_mb\"], dc.get(\"chunk\",{}).get(\"max_pct_range\"), dc.get(\"vs_host_chain\",{}).get(\"identical\"), {k:v.get(\"max_pct_range\") for k,v in fp.get(\"reports\",{}).items() if isinstance(v,dict)})"
}
run n16-old groot-n16-base groot-n16-base-split groot-trt
run n16-h16 groot-n16-base groot-n16-base-split-h16 groot-trt-h16
run n17-old groot-n17-base groot-n17-base-split groot-n17-base-trt
run n17-h16 groot-n17-base groot-n17-base-split-h16 groot-n17-base-trt-h16
echo GROOT_DONE
