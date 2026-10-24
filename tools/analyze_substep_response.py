"""Read-only comparison of cycle substep probes; no changed acceptance threshold."""
import argparse
import json
import pathlib
import numpy as np

p=argparse.ArgumentParser();p.add_argument('batch_dir',type=pathlib.Path);a=p.parse_args()
baseline=json.loads((a.batch_dir/'baseline/diagnostic_trace.json').read_text())
report={}
for folder in sorted(a.batch_dir.iterdir()):
    if not (folder/'substeps.npz').exists():continue
    rows=json.loads((folder/'diagnostic_trace.json').read_text())
    result=json.loads((folder/'diagnostic_evaluation.json').read_text())
    data=np.load(folder/'substeps.npz')
    array=lambda k:np.asarray([r[k] for r in rows])
    phase=array('phase');clock=array('phase_time');valid=array('valid');raw=array('raw')
    mask=valid&(phase==2)&(clock>.5)
    count=mask.sum(0);tolerance=np.array([.005,.015])+.25*np.abs(raw)
    instant=np.stack((array('forward'),array('heading_rate')),-1)
    physical=data['interval'][:,0,:,:2];fluctuation=data['interval'][:,1,:,:2]
    stats={}
    for label,values in [('endpoint',instant),('physics_interval_mean',physical)]:
        good=np.abs(values-raw)<tolerance
        stats[label]=dict(mae=np.abs(values-raw)[mask].mean(0).tolist(),
            axis_pass=(((good&mask[:,:,None]).sum(0)/np.maximum(count[:,None],1)>=.8)&(count[:,None]>=20)).sum(0).tolist(),
            joint_pass=int((((good.all(-1)&mask).sum(0)/np.maximum(count,1)>=.8)&(count>=20)).sum()))
    n=min(len(rows),len(baseline));pre=np.asarray([r['valid'] for r in baseline[:n]])&(np.asarray([r['phase'] for r in baseline[:n]])<2)
    # Include phase2 before the first intervention. Once an environment crossed
    # intervention, no later landing/reset interval may count as "pre".
    seen=np.maximum.accumulate((np.asarray([r['phase'] for r in baseline[:n]])==2)&(np.asarray([r['phase_time'] for r in baseline[:n]])>=.6),axis=0)
    pre=np.asarray([r['valid'] for r in baseline[:n]])&~seen
    deviations={}
    for key in ['raw','applied','pos','heading_rate','front_z']:
        old=np.asarray([r[key] for r in baseline[:n]]);new=array(key)[:n]
        deviations[key]=float(np.abs(old-new)[pre].max())
    stats.update(complete=result['complete'],qualified=result['qualified'],pre_intervention_max=deviations,
        mean_within_interval_std=fluctuation[mask].mean(0).tolist(),
        endpoint_minus_physics_mean_mae=np.abs(instant-physical)[mask].mean(0).tolist())
    # First four environments have all 2ms samples, not only moments.
    x=data['selected'][...,:2];m=mask[:,:4];err=np.abs(x-raw[:,:4][:,None])
    stats['selected_physics_good_fraction']=((err<tolerance[:,:4][:,None])*m[:,None,:,None]).sum((0,1,2)).tolist()
    denom=int(m.sum())*x.shape[1]
    stats['selected_physics_good_fraction']=[v/max(1,denom) for v in stats['selected_physics_good_fraction']]
    report[folder.name]=stats
(a.batch_dir/'substep_response.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
