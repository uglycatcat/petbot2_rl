"""Compare actual trajectories without adding posture or heading success conditions."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(evaluation):
    evaluation=Path(evaluation)
    result=json.loads((evaluation/'evaluation.json').read_text())
    trace=np.load(evaluation/'evaluation_traces.npz')
    dt=result['trace_dt']; names=result['joint_names']
    angle_ids=[names.index(n) for n in ('head','left_frame','right_frame')]
    rows=[]
    for i in range(result['episodes']):
        valid=trace['alive'][:,i].astype(bool)
        q=trace['joint_pos'][valid,i][:,angle_ids]
        qd=trace['joint_vel'][valid,i][:,angle_ids]
        target=trace['targets'][valid,i]
        bounded=np.concatenate((target[:,:1]/1.3,
            np.where(target[:,1:3]>=0,target[:,1:3]/1.2,target[:,1:3]),target[:,3:]/25.),axis=1)
        rate=np.diff(np.vstack((np.zeros((1,5)),bounded)),axis=0)/dt
        wheel=trace['wheel_pos'][valid,i];bottom=trace['bottom'][valid,i]
        top=trace['top'][valid,i];root=trace['root_pose'][valid,i]
        far=(result['step_edge']-wheel[:,:,0].max(-1))>.20
        w,x,y,z=root[:,3:].T
        roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
        pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
        yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        first=[float((np.flatnonzero(top[:,j])[0]+1)*dt) if top[:,j].any() else None for j in range(4)]
        duration=len(q)*dt
        success=bool(trace['success'][valid,i].any())
        descriptor=[float(yaw[-1]/np.pi),float(np.abs(roll).max()/np.pi),
            float(np.abs(pitch).max()/np.pi),float((q[:,1]-q[:,2]).mean()/2.2),
            float(q[:,0].mean()/1.3),float(root[-1,1]/.5),
            *[t/duration if t is not None else 1. for t in first],
            float(np.mean(np.diff(root[:,0])<-.001)) if len(q)>1 else 0.]
        rows.append({'episode':i,'success':success,'duration_s':duration,
            'head_abs_mean_rad':float(np.abs(q[:,0]).mean()),
            'head_final_rad':float(q[-1,0]),
            'far_frame_abs_mean_rad':float(np.abs(q[far,1:3]).mean()) if far.any() else None,
            'far_wheel_lift_fraction':float((bottom[far]>.015).any(-1).mean()) if far.any() else None,
            'far_duration_s':float(far.sum()*dt),
            'bounded_target_rate_rms_per_s':float(np.sqrt(np.mean(rate**2))),
            'angle_speed_rms_rad_s':float(np.sqrt(np.mean(qd**2))),
            'angle_acceleration_rms_rad_s2':float(np.sqrt(np.mean((np.diff(qd,axis=0)/dt)**2))) if len(qd)>1 else 0.,
            'max_abs_roll_rad':float(np.abs(roll).max()),'max_abs_pitch_rad':float(np.abs(pitch).max()),
            'final_yaw_rad':float(yaw[-1]),'final_lateral_m':float(root[-1,1]),
            'first_top_time_s_LF_RF_LR_RR':first,'max_simultaneous_top':int(top.sum(-1).max()),
            'descriptor':descriptor})
    success_rows=[r for r in rows if r['success']]
    def aggregate(items):
        output={}
        for key in ('duration_s','head_abs_mean_rad','far_frame_abs_mean_rad','far_wheel_lift_fraction',
                    'bounded_target_rate_rms_per_s','angle_speed_rms_rad_s','angle_acceleration_rms_rad_s2'):
            vals=[r[key] for r in items if r[key] is not None]
            output[key]={'mean':float(np.mean(vals)),'min':float(np.min(vals)),'max':float(np.max(vals))} if vals else None
        return output
    return {'result':result,'all_episodes':aggregate(rows),'successful_episodes':aggregate(success_rows),
        'behavior_descriptor':np.mean([r['descriptor'] for r in success_rows],axis=0).tolist() if success_rows else None,
        'descriptor_definition':['final_yaw/pi','max_abs_roll/pi','max_abs_pitch/pi','mean_frame_difference/2.2',
            'mean_head/1.3','final_lateral/.5','first_top_LF/duration','first_top_RF/duration',
            'first_top_LR/duration','first_top_RR/duration','backward_step_fraction'],
        'scope':'Fixed default initial state; far means nearest wheel is over20cm from edge. Metrics are diagnostic, not success gates.',
        'episodes':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=summarize(a.evaluation);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'successes':data['result']['successes'],'metrics':data['successful_episodes']}))
