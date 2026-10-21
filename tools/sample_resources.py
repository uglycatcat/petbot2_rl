"""Sample whole-card load; memory occupancy is never presented as compute utilization."""
import argparse,json,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--seconds',type=int,default=60);a=p.parse_args()
rows=[];deadline=time.monotonic()+a.seconds
while time.monotonic()<deadline:
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 rows.append({'time':time.time(),'gpus':[list(map(int,row.split(','))) for row in raw.splitlines()]});time.sleep(2)
summary={}
for gpu in range(4):
 x=[r['gpus'][gpu] for r in rows];summary[gpu]={'mean_compute_percent':sum(v[3] for v in x)/len(x),'peak_memory_mib':max(v[2] for v in x),'total_memory_mib':x[0][1]}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'samples':rows,'summary':summary},indent=2));print(json.dumps(summary))
