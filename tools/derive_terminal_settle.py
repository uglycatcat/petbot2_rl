"""Derive an explicit neutral-posture suffix after observed four-wheel support."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from recover.contract import POSITION_LOW,POSITION_HIGH,CONTROL_DT
from recover.model import source_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--witness',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True)
    p.add_argument('--ramp-seconds',type=float,default=.5);p.add_argument('--support-seconds',type=float,default=.1)
    args=p.parse_args();w=Path(args.witness);evaluation=Path(args.evaluation);out=Path(args.output)
    m=json.loads(w.with_suffix('.json').read_text());r=json.loads(evaluation.with_suffix('.json').read_text())
    if m['source_hash']!=source_fingerprint() or r['source_hash']!=m['source_hash']:raise ValueError('Source differs')
    if r.get('pose_ids')!=[m['pose_id']]:raise ValueError('Witness and evaluation initial poses differ')
    if r['backend']!='original_mujoco':raise ValueError('This derivation requires native ground-only contact evidence')
    archive=np.load(w.with_suffix('.npz'));a=archive['actions'];trace=np.load(evaluation.with_suffix('.npz'))
    if not np.array_equal(trace['action'][:len(a),0],a):raise ValueError('Evaluation does not replay these exact targets')
    if hashlib.sha256(a.tobytes()).hexdigest()!=m['action_sha256']:raise ValueError('Witness checksum mismatch')
    support=round(args.support_seconds/CONTROL_DT);ramp=round(args.ramp_seconds/CONTROL_DT)
    if support<1 or ramp<1 or abs(support*CONTROL_DT-args.support_seconds)>1e-8 or abs(ramp*CONTROL_DT-args.ramp_seconds)>1e-8:
        raise ValueError('Durations must be positive multiples of the control period')
    bodies=r['body_names'];wheel_ids=[bodies.index(n) for n in ('left_front_wheel','right_front_wheel','left_rear_wheel','right_rear_wheel')]
    ground=trace['ground_force'][:len(a),0]
    contact=(ground[:,:,wheel_ids,2]>.05).mean(1)>=.8
    eligible=trace['eligible'][:len(a),0]
    four=contact.all(1)&eligible&(trace['up'][:len(a),0]>0)
    candidates=np.flatnonzero(np.convolve(four.astype(int),np.ones(support,dtype=int),mode='valid')==support)
    if not len(candidates):raise ValueError('No sustained four-wheel ground support in the recorded sequence')
    count=int(candidates[0]+support)
    if count+ramp>len(a):raise ValueError('The suffix would exceed the original recorded horizon')
    neutral=np.zeros(7,dtype=np.float32);neutral[:5]=2*(0-np.array(POSITION_LOW))/(np.array(POSITION_HIGH)-np.array(POSITION_LOW))-1
    actions=a.copy();weight=np.arange(1,ramp+1,dtype=np.float32)/ramp
    actions[count:count+ramp]=a[count-1]+weight[:,None]*(neutral-a[count-1]);actions[count+ramp:]=neutral
    actions[count:,5:]=0
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.with_suffix('.npz').exists() or out.with_suffix('.json').exists():raise ValueError('Never overwrite a candidate')
    np.savez_compressed(out.with_suffix('.npz'),actions=actions,initial_root=archive['initial_root'],initial_joint=archive['initial_joint'])
    m.update(status='engineered_terminal_suffix_requires_independent_replay',action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        derived_from=str(w),derivation=dict(method='after_sustained_ground_four_ramp_all_position_targets_to_zero_and_brake',
        evidence=str(evaluation),trigger_s=count*CONTROL_DT,support_seconds=args.support_seconds,ramp_seconds=args.ramp_seconds,
        source_action_sha256=hashlib.sha256(a.tobytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='The recovery prefix is searched; this neutral-posture suffix is explicitly engineered, not claimed to be learned.'))
    out.with_suffix('.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m['derivation']),flush=True)


if __name__=='__main__':main()
