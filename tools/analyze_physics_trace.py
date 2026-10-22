"""Summarize measured rear-wheel force gaps, not geometric separation or success."""
import argparse,json,pathlib,numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=pathlib.Path);a=p.parse_args()
d=np.load(a.directory/'physics_trace.npz');m=json.loads((a.directory/'replay_meta.json').read_text());dt=m['physics_dt'];valid=d['valid']&(d['since']>=2.)
# Invalid/reset/transition intervals break runs, never concatenate different episodes.
def runs(mask):
 result=[]
 for row in mask:
  padded=np.r_[False,row,False].astype(np.int8);diff=np.diff(padded)
  result.extend((np.where(diff==-1)[0]-np.where(diff==1)[0]).tolist())
 return np.array(result,dtype=float)*dt*1000
summary={}
for threshold in [0.,.05]:
 contact=d['force']>threshold
 for label,missing in [('either_rear',~contact.all(-1)),('both_rear',~contact.any(-1))]:
  values=runs((missing&valid[:,:,None]).transpose(1,0,2).reshape(valid.shape[1],-1))
  summary[f'{label}_force_le_{threshold}']={'events':len(values),'p50_ms':float(np.median(values)) if len(values) else 0.,'p95_ms':float(np.quantile(values,.95)) if len(values) else 0.,'max_ms':float(values.max()) if len(values) else 0.,'fraction':float((missing&valid[:,:,None]).sum()/max(1,valid.sum()*missing.shape[-1]))}
summary['physics_dt']=dt;summary['control_dt']=m['control_dt'];summary['note']='Force below threshold; not proof of airborne state. First 2 active seconds excluded.'
(a.directory/'physical_gap_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
