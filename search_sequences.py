"""Scratch episode-level CEM policy search, complementary to feedback PPO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from recover.contract import CONTROL_DT,POSITION_LOW,POSITION_HIGH
from recover.model import (POSITION_JOINTS,JOINT_ORDER,source_fingerprint,
                           task_implementation_fingerprint)
from recover.poses import load_poses
from recover.sequence_native import NativeSequences

ACTIVE_OUTPUT=None

class ResourceInterruption(RuntimeError):
    """A forced resource stop; the last complete generation remains resumable."""

def expand_targets(parameters, initial_targets, wheel_mode, steps=800, blend_steps=10):
    """Equal-duration target holds, with short linear transitions; no support script."""
    population,knots,dimensions=parameters.shape
    targets=np.zeros((population,knots,7),dtype=np.float32)
    targets[:,:,:dimensions]=np.tanh(parameters).astype(np.float32)
    actions=np.zeros((population,steps,7),dtype=np.float32)
    boundaries=np.linspace(0,steps,knots+1).round().astype(int)
    for k in range(knots):
        begin,end=boundaries[k:k+2];actions[:,begin:end]=targets[:,k,None]
        previous=initial_targets if k==0 else targets[:,k-1]
        duration=min(blend_steps,end-begin)
        if duration:
            weight=np.arange(1,duration+1,dtype=np.float32)/duration
            actions[:,begin:begin+duration]=previous[:,None]+weight[None,:,None]*(targets[:,k,None]-previous[:,None])
    if wheel_mode=='zero':actions[:,:,5:]=0
    else:
        # One second of zero drive provides a generic chance to stop, regardless of posture.
        actions[:,-50:,5:]=0
    return actions


def save_witness(prefix,actions,pose,poses,addresses,placement,metadata):
    root=np.zeros(13,dtype=np.float32);root[:7]=poses[pose,:7]
    joints=np.array([poses[pose,addresses[n]] for n in JOINT_ORDER],dtype=np.float32)
    np.savez_compressed(prefix.with_suffix('.npz'),actions=actions,initial_root=root,initial_joint=joints)
    description=dict(metadata,pose_id=f'pose_{pose+1:02d}',steps=len(actions),control_dt=CONTROL_DT,
                     action_semantics='applied_absolute_normalized_targets',robot_joint_names=list(JOINT_ORDER),
                     placement=placement[pose],action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                     status='search_candidate_requires_independent_replay')
    prefix.with_suffix('.json').write_text(json.dumps(description,indent=2)+'\n')


def main(engine_factory=NativeSequences):
    global ACTIVE_OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--wheel-mode',choices=['zero','free'],required=True)
    parser.add_argument('--population-per-pose',type=int,default=32)
    parser.add_argument('--generations',type=int,default=40)
    parser.add_argument('--knots',type=int,default=8)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true',help='Only resume the same forcibly interrupted run')
    args=parser.parse_args()
    if args.population_per_pose<8 or args.generations<1 or not 2<=args.knots<=32:parser.error('Invalid search dimensions')
    output=Path(args.output)
    if output.exists() and not args.resume:raise ValueError('A fresh search requires a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    source_hash=source_fingerprint();code_hash=task_implementation_fingerprint()
    poses,addresses,placement=load_poses();population=args.population_per_pose;dimensions=5 if args.wheel_mode=='zero' else 7
    low=np.array(POSITION_LOW);high=np.array(POSITION_HIGH)
    initial_targets=np.zeros((4,7),dtype=np.float32)
    initial_targets[:,:5]=2*(poses[:,[addresses[n] for n in POSITION_JOINTS]]-low)/(high-low)-1
    # Recorded soft-limit overshoot is never written as a target beyond the action bounds.
    initial_targets=np.clip(initial_targets,-1,1)
    rng=np.random.default_rng(args.seed)
    mean=np.zeros((4,args.knots,dimensions));std=np.full_like(mean,1.5)
    best=np.full(4,-np.inf);found=np.zeros(4,dtype=bool);first_generation=0
    metadata=dict(source_hash=source_hash,code_hash=code_hash,seed=args.seed,wheel_mode=args.wheel_mode,
                  method='cross_entropy_episode_policy_search',args=vars(args),initialization='scratch_no_demo_or_weights',
                  backend=engine_factory.backend,physics_dt=engine_factory.physics_dt)
    if args.resume:
        if (output/'failure.json').exists():raise ValueError('A failed search is not eligible for forced-interruption recovery')
        previous=json.loads((output/'run.json').read_text());status=json.loads((output/'status.json').read_text())
        if status['status'] not in ('running','resource_stopped'):raise ValueError('Only a forcibly interrupted search may resume')
        if previous['source_hash']!=source_hash or previous['code_hash']!=code_hash:raise ValueError('Resume source differs')
        if previous.get('backend')!=engine_factory.backend:raise ValueError('Resume backend differs')
        expected=dict(previous['args'],resume=True)
        if expected!=vars(args):raise ValueError('Resume configuration differs')
        if (output/'state.npz').exists():
            checkpoint=np.load(output/'state.npz');mean=checkpoint['mean'];std=checkpoint['std'];best=checkpoint['best'];found=checkpoint['found']
            first_generation=int(checkpoint['generation'])+1;rng.bit_generator.state=json.loads(str(checkpoint['rng']))
    else:(output/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    ACTIVE_OUTPUT=output
    (output/'status.json').write_text(json.dumps(dict(status='running',pid=os.getpid(),generation=first_generation,
                                                     total_generations=args.generations),indent=2)+'\n')
    start=time.monotonic();elite_count=max(4,population//4)
    with engine_factory(threads=args.threads,seed=args.seed) as engine:
        metadata['backend']=engine.backend;metadata['physics_dt']=engine.physics_dt
        for generation in range(first_generation,args.generations):
            control=output.parent/'control.json'
            if control.exists() and not json.loads(control.read_text()).get('enabled',False):
                status=dict(status='control_stopped',pid=os.getpid(),generation=generation,
                            total_generations=args.generations,success_found=found.tolist())
                (output/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
                return
            available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
            if available<2*1024*1024:
                status=dict(status='resource_stopped',pid=os.getpid(),generation=generation,
                            total_generations=args.generations,reason='Host available memory below 2 GiB',
                            success_found=found.tolist())
                (output/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
                return
            parameters=mean[:,None]+std[:,None]*rng.standard_normal((4,population,args.knots,dimensions))
            # Independent broad draws retain routes that a single elite family would lose.
            immigrants=max(2,population//4)
            parameters[:,-immigrants:]=1.5*rng.standard_normal((4,immigrants,args.knots,dimensions))
            parameters[:,0]=mean
            # This is the only action schedule: random goals, generic interpolation, optional zero drive.
            actions=np.concatenate([expand_targets(parameters[p],np.repeat(initial_targets[p,None],population,axis=0),args.wheel_mode) for p in range(4)])
            state=np.repeat(poses,population,axis=0)
            try:summary,_=engine.run(state,actions)
            except ResourceInterruption as error:
                status=dict(status='resource_stopped',pid=os.getpid(),generation=generation,
                            total_generations=args.generations,reason=str(error),success_found=found.tolist())
                (output/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
                return
            summary=summary.reshape(4,population,10)
            for pose in range(4):
                rank=np.argsort(summary[pose,:,0])[::-1];elite=parameters[pose,rank[:elite_count]]
                mean[pose]=.4*mean[pose]+.6*elite.mean(0)
                std[pose]=np.maximum(.2,.4*std[pose]+.6*elite.std(0))
                winner=int(rank[0]);score=float(summary[pose,winner,0])
                if score>best[pose]:
                    best[pose]=score
                    prefix=output/f'best_pose_{pose+1:02d}'
                    save_witness(prefix,actions[pose*population+winner],pose,poses,addresses,placement,
                                 dict(metadata,generation=generation,kernel_sha256=engine.kernel_sha256,metrics=summary[pose,winner].tolist()))
                winners=np.flatnonzero(summary[pose,:,1]>=0)
                if len(winners) and not found[pose]:
                    winner=int(winners[np.argmax(summary[pose,winners,0])]);found[pose]=True
                    save_witness(output/f'first_success_pose_{pose+1:02d}',actions[pose*population+winner],pose,poses,addresses,placement,
                                 dict(metadata,generation=generation,kernel_sha256=engine.kernel_sha256,metrics=summary[pose,winner].tolist()))
            # State checkpoint and RNG round-trip are also used by the short engineering gate.
            temporary=output/'state.tmp.npz'
            np.savez_compressed(temporary,mean=mean,std=std,best=best,found=found,generation=generation,rng=json.dumps(rng.bit_generator.state))
            temporary.replace(output/'state.npz')
            reloaded=np.load(output/'state.npz');check=np.random.default_rng();check.bit_generator.state=json.loads(str(reloaded['rng']))
            probe=np.random.default_rng();probe.bit_generator.state=rng.bit_generator.state
            assert np.array_equal(check.standard_normal(16),probe.standard_normal(16)) and np.array_equal(mean,reloaded['mean'])
            status=dict(status='running',pid=os.getpid(),generation=generation+1,total_generations=args.generations,
                        episodes_per_pose=(generation+1)*population,seconds=time.monotonic()-start,
                        best_score=best.tolist(),success_found=found.tolist(),generation_successes=(summary[:,:,1]>=0).sum(1).tolist(),
                        generation_max_up=summary[:,:,3].max(1).tolist(),generation_max_hold=summary[:,:,2].max(1).tolist(),
                        generation_four_steps=summary[:,:,4].sum(1).tolist(),invalid_count=(summary[:,:,5]>=0).sum(1).tolist())
            (output/'status.tmp.json').write_text(json.dumps(status,indent=2)+'\n');(output/'status.tmp.json').replace(output/'status.json')
            with (output/'metrics.jsonl').open('a') as log:log.write(json.dumps(status)+'\n')
            print(json.dumps(status),flush=True)
            if found.all():break
    status['status']='review_required';(output/'status.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except Exception as error:
        if ACTIVE_OUTPUT is not None:
            failure=dict(status='failed',error=repr(error))
            (ACTIVE_OUTPUT/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
            (ACTIVE_OUTPUT/'status.json').write_text(json.dumps(failure,indent=2)+'\n')
        raise
