"""Describe support and actuator evidence without automatically declaring a new mechanism."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from recover.model import load_model,source_fingerprint,REAR_WHEELS


def analyze(prefix,output):
    report=json.loads(prefix.with_suffix('.json').read_text())
    if report.get('backend')!='original_mujoco': raise ValueError('Use an original-MuJoCo evaluation')
    if report['source_hash']!=source_fingerprint(): raise ValueError('The source model differs from the evaluation')
    z=np.load(prefix.with_suffix('.npz'));dt=report['control_dt']
    if 'ground_force' not in z:
        raise ValueError('Re-evaluate with ground_force logging; net body contact forces also contain self contacts')
    bodies=report['body_names'];names=list(load_model()['actuators'])
    rear_act=[names.index(n) for n in REAR_WHEELS]
    wheel_names=['left_front_wheel','right_front_wheel',*REAR_WHEELS]
    nonwheel=[i for i,n in enumerate(bodies) if n not in ['world',*wheel_names]]
    rows=[];poses=[]
    pose_ids=report.get('pose_ids',[f'pose_{i+1:02d}' for i in range(len(report['success']))])
    for pose,pose_id in enumerate(pose_ids):
        count=len(z['up']);time=np.arange(1,count+1)*dt
        valid=z['eligible'][:,pose] if 'eligible' in z else np.ones(count,dtype=bool)
        # Mechanism evidence must exclude wheel self-contacts. The task's net
        # contact mask is retained in the trace, but is not a ground-support mask.
        wheel_ids=[bodies.index(n) for n in wheel_names]
        wheel_ground=np.take(z['ground_force'][:,pose,:,:,2],wheel_ids,axis=2)
        contact=(wheel_ground>.05).mean(1)>=.8
        upright=(z['up'][:,pose]>np.cos(np.deg2rad(25))) & valid
        hits=np.flatnonzero(upright)
        stable=(z['hold'][:,pose]>=.5)&valid
        if bool(stable.any())!=bool(report['success'][pose]):
            raise ValueError('Reported success disagrees with the recorded stable hold')
        peak=int(np.argmax(np.where(valid,z['up'][:,pose],-np.inf)))
        end=int(hits[0]) if len(hits) else peak
        interval=slice(0,end+1)
        support_codes=(contact*np.array([1,2,4,8])).sum(-1)
        # Mean upward force across all physics samples; negative forces are not support.
        vertical=np.maximum(z['ground_force'][:,pose,:,:,2],0).mean(1)
        other_support=vertical[:,nonwheel].sum(-1)
        total_support=vertical[:,[bodies.index(n) for n in wheel_names]].sum(-1)+other_support
        both_rear=contact[:,2:].all(-1)
        only_rear=both_rear & ~contact[:,:2].any(-1)
        sequence=[]
        for t,code in enumerate(support_codes[:end+1]):
            if not sequence or sequence[-1]['wheel_mask']!=int(code):
                sequence.append(dict(start_s=round(t*dt,4),end_s=round(float(time[t]),4),wheel_mask=int(code)))
            else: sequence[-1]['end_s']=round(float(time[t]),4)
        torque=None;power=None
        if 'effort' in z:
            torque=z['effort'][:,pose,rear_act]
            power=(torque*z['wheel'][:,pose]).sum(-1)
        summary=dict(pose_id=pose_id,success=bool(report['success'][pose]),
            first_success_s=report['first_success_s'][pose],
            first_upright_s=float(time[hits[0]]) if len(hits) else None,
            peak_up=float(z['up'][peak,pose]),peak_up_s=float(time[peak]),
            described_interval_end_s=float(time[end]),
            described_interval='start_to_first_upright' if len(hits) else 'start_to_best_uprightness',
            both_rear_contact_s=float(both_rear[interval].sum()*dt),
            rear_only_contact_s=float(only_rear[interval].sum()*dt),
            nonwheel_support_s=float((other_support[interval]>.05).sum()*dt),
            nonwheel_support_by_body_s={bodies[i]:float((vertical[interval,i]>.05).sum()*dt) for i in nonwheel},
            wheel_support_sequence=sequence,
            rear_peak_abs_torque_nm=np.abs(torque[interval]).max(0).tolist() if torque is not None else None,
            rear_positive_work_sample_estimate_j=float(np.maximum(power[interval],0).sum()*dt) if power is not None else None,
            conclusion='recovery_witness_pending_mechanism_review' if stable.any() else 'no_recovery_witness')
        poses.append(summary)
        for t in range(count):
            rows.append(dict(pose_id=summary['pose_id'],time_s=round(float(time[t]),4),eligible=bool(valid[t]),
                up=float(z['up'][t,pose]),wheel_mask=int(support_codes[t]),stable_hold_s=float(z['hold'][t,pose]),
                nonwheel_upward_force_n=float(other_support[t]),total_upward_force_n=float(total_support[t]),
                rear_left_speed_rad_s=float(z['wheel'][t,pose,0]),rear_right_speed_rad_s=float(z['wheel'][t,pose,1]),
                rear_sampled_motor_power_w=float(power[t]) if power is not None else None))
    result=dict(evaluation=str(prefix),source_hash=report['source_hash'],poses=poses,
        contact_basis='Ground-only upward force > 0.05 N in at least 80% of physical substeps',
        wheel_mask_bits=dict(zip(wheel_names,[1,2,4,8])),
        interpretation='Support durations and sampled motor work describe evidence, not causation or novelty. '
                       'Motor work uses one actuator-force sample per control step and may miss substep peaks. '
                       'Review full trajectories/video and controlled comparisons before identifying a new mechanism.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    with output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps([dict(pose=p['pose_id'],success=p['success'],rear_support_s=p['both_rear_contact_s'],
                           nonwheel_support_s=p['nonwheel_support_s']) for p in poses]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();analyze(Path(args.evaluation),Path(args.output))
