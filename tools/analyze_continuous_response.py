"""Offline command-response diagnosis. Windowed metrics never replace acceptance."""
import argparse
import collections
import json
import pathlib
import numpy as np


def analyze(folder):
    prefix='' if (folder/'evaluation.json').exists() else 'diagnostic_'
    rows=json.loads((folder/(prefix+'trace.json')).read_text())
    result=json.loads((folder/(prefix+'evaluation.json')).read_text())
    a=lambda key:np.asarray([r[key] for r in rows])
    raw=a('raw');valid=a('valid')&(a('phase')==2)&(a('phase_time')>.5)
    actual=np.stack((a('forward'),a('heading_rate')),-1)
    measures={'instant':actual,'ema120ms':a('motion_feedback')}
    interval=a('interval_motion')
    for width in [5,17]:
        prefix=np.concatenate((np.zeros_like(interval[:1]),np.cumsum(interval,axis=0)),axis=0)
        end=np.arange(1,len(rows)+1);start=np.maximum(0,end-width)
        measures[f'pose_window_{width}steps']=(prefix[end]-prefix[start])/(end-start)[:,None,None]
    age=np.zeros(valid.shape);dt=float(rows[1]['t']-rows[0]['t'])
    for t in range(1,len(rows)):
        age[t]=np.where((raw[t]==raw[t-1]).all(-1),age[t-1]+dt,0.)
    tolerance=np.array([.005,.015])+.25*np.abs(raw)
    report={}
    for period,mask in [('all',valid),('settled',valid&(age>=.6))]:
        counts=mask.sum(0);eligible=counts>=20
        report[period]={}
        for name,value in measures.items():
            error=value-raw;good=np.abs(error)<tolerance
            fractions=(good*mask[:,:,None]).sum(0)/np.maximum(counts[:,None],1)
            joint=((good.all(-1)&mask).sum(0)/np.maximum(counts,1)>=.8)&eligible
            report[period][name]=dict(count=int(mask.sum()),episode_count=int(eligible.sum()),
                axis_pass=((fractions>=.8)&eligible[:,None]).sum(0).tolist(),joint_pass=int(joint.sum()),
                mae=np.abs(error)[mask].mean(0).tolist(),bias=error[mask].mean(0).tolist(),
                good_fraction=good[mask].mean(0).tolist())
    v,w=raw[...,0],raw[...,1]
    categories=dict(forward=(v>0)&(w==0),reverse=(v<0)&(w==0),left=(v==0)&(w>0),
                    right=(v==0)&(w<0),curve=(v!=0)&(w!=0),stop=(v==0)&(w==0))
    applied=a('applied');half=(applied[...,2]-applied[...,1])*12.5
    report['categories']={}
    for name,category in categories.items():
        mask=valid&(age>=.6)&category
        if not mask.any():continue
        report['categories'][name]=dict(samples=int(mask.sum()),command=raw[mask].mean(0).tolist(),
            instantaneous=actual[mask].mean(0).tolist(),motion=measures['ema120ms'][mask].mean(0).tolist(),
            actual_std=actual[mask].std(0).tolist(),half_difference_mean=float(half[mask].mean()),
            half_difference_std=float(half[mask].std()),
            abs_error=np.abs(actual-raw)[mask].mean(0).tolist(),filtered_abs_error=np.abs(measures['ema120ms']-raw)[mask].mean(0).tolist())
    codes=a('failure_code');mask=a('valid')&(codes!=0)
    report['failure_codes']=dict(collections.Counter(map(str,codes[mask].tolist())))
    report['complete']=result['complete']
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    result={str(folder):analyze(folder) for folder in args.folders}
    args.output.write_text(json.dumps(result,indent=2));print('Wrote',len(result),'diagnoses to',args.output)
