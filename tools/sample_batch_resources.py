"""One finite resource sample after batch deployment; never launches training."""
import argparse
import json
import pathlib
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('batch_dir',type=pathlib.Path);p.add_argument('--samples',type=int,default=30);p.add_argument('--interval',type=float,default=2.);a=p.parse_args()
assert 1<=a.samples<=60 and 0<=a.interval<=5
rows=[]
for i in range(a.samples):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
    rows.append(dict(time=time.strftime('%FT%T%z'),gpus=[list(map(int,row.split(','))) for row in raw.splitlines()]))
    if i<a.samples-1:time.sleep(a.interval)
summary={}
for gpu in [row[0] for row in rows[0]['gpus']]:
    values=[next(v for v in row['gpus'] if v[0]==gpu) for row in rows]
    summary[gpu]=dict(total_mib=values[0][1],peak_used_mib=max(v[2] for v in values),mean_utilization=sum(v[3] for v in values)/len(values),min_utilization=min(v[3] for v in values),max_utilization=max(v[3] for v in values))
(a.batch_dir/'deployed_resource_samples.json').write_text(json.dumps(dict(samples=rows,summary=summary),indent=2))
print(json.dumps(summary))
