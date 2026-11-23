"""Read existing physical evidence; no simulation, training or running-input changes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(directory):
    result=json.loads((directory/'evaluation.json').read_text())
    data=np.load(directory/'evaluation_traces.npz')
    assert all(np.isfinite(data[k]).all() for k in data.files)
    pos=data['wheel_pos'];bottom=data['bottom'];force=data['forces'];valid=data['alive']
    physical=(pos[...,0]>=result['step_edge']+.003)&(pos[...,0]<=result['step_edge']+2.-.003)
    physical&=(np.abs(pos[...,1])<=1.-.003)&(np.abs(bottom-result['step_height'])<=.004)&(force[...,2]>.03)
    successful=(physical.all(-1)&valid).any(0)
    assert successful.sum()==result['successes']
    assert np.array_equal(successful,(data['success']&valid).any(0))
    targets=data['targets'][valid]
    assert np.abs(targets[:,0]).max()<=1.300001
    assert targets[:,1:3].min()>=-1.000001 and targets[:,1:3].max()<=1.200001
    assert np.abs(targets[:,3:]).max()<=25.000001
    quat=data['root_pose'][...,3:];up=1.-2.*(quat[...,1]**2+quat[...,2]**2)
    min_up=np.where(valid,up,1.).min(0)
    output={'result':result,'all_finite':True,'targets_within_limits':True,
        'target_min':targets.min(0).tolist(),'target_max':targets.max(0).tolist(),
        'measured_joint_position_min':data['joint_pos'][valid].min(0).tolist(),
        'measured_joint_position_max':data['joint_pos'][valid].max(0).tolist(),
        'measured_joint_speed_abs_max':np.abs(data['joint_vel'][valid]).max(0).tolist(),
        'body_up_min_range':[float(min_up.min()),float(min_up.max())],
        'episodes_body_up_below_zero':int((min_up<0).sum()),
        'episodes_body_up_below_minus_half':int((min_up<-.5).sum()),
        'peak_simultaneous_wheels':np.where(valid,physical.sum(-1),0).max(0).tolist()}
    if successful.any():
        ids=np.flatnonzero(successful);events=np.argmax(physical.all(-1)&valid,axis=0)[ids]
        output['success_physics']={'maximum_bottom_error_m':float(np.abs(bottom[events,ids]-result['step_height']).max()),
            'minimum_support_N':float(force[events,ids,:,2].min())}
    failures=np.flatnonzero(~successful)
    if len(failures):
        count=np.where(valid,physical.sum(-1),0)
        episode=int(failures[np.argmax(count[:,failures].max(0))]);frame=int(np.argmax(count[:,episode]))
        output['representative_failure']={'episode':episode,'time_s':(frame+1)*result['trace_dt'],
            'top_LF_RF_LR_RR':physical[frame,episode].tolist(),
            'wheel_position':pos[frame,episode].tolist(),'wheel_bottom':bottom[frame,episode].tolist(),
            'wheel_support_N':force[frame,episode,:,2].tolist(),'root_pose':data['root_pose'][frame,episode].tolist()}
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    records=json.loads((a.batch/'catalog.json').read_text())
    snapshot=[]
    for record in records:
        trial=Path(record['evaluation']).parent
        directory=trial/'export_eval64' if (trial/'export_eval64/evaluation.json').exists() else Path(record['evaluation'])
        checked=audit(directory)
        metrics=json.loads((directory/'motion_metrics.json').read_text())
        tb=json.loads((directory/'tensorboard_review.json').read_text())
        chosen=Path(record['selected_checkpoint'])
        assert hashlib.sha256(chosen.read_bytes()).hexdigest()==record['checkpoint_sha256']
        snapshot.append({'trial':str(trial),'record':record,'physical_audit':checked,
            'motion_metrics':metrics['successful_episodes'],
            'training_selected_curves':{k:v for k,v in tb.items() if k.startswith('Task/') or k.startswith('MotionCost/') or k=='Policy/mean_noise_std'}})
    result={'batch':str(a.batch),'reviewed_trials':len(snapshot),'trials':snapshot,
        'note':'Independent post hoc audit of frozen recorded trajectories; source model and active runs unmodified.'}
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    for entry in snapshot:
        r=entry['record'];q=entry['physical_audit']
        print(entry['trial'].replace(str(a.batch)+'/',''),r['step_height'],r['success_rate'],
              'deep_inversion',q['episodes_body_up_below_minus_half'],'up',q['body_up_min_range'])


if __name__=='__main__':main()
