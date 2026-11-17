"""Diagnostic replay of four fixed absolute-target sequences in the training engine."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sequences',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--num-envs',type=int,default=64)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();app=AppLauncher(args).app
try:
    import numpy as np
    import torch
    from recover.env import RecoverEnv,EnvCfg
    from recover.model import source_fingerprint,task_implementation_fingerprint
    from recover.fidelity import check_initial_body_poses
    prefix=Path(args.sequences);metadata=json.loads(prefix.with_suffix('.json').read_text())
    if metadata['source_hash']!=source_fingerprint():raise ValueError('The source model differs from the sequences')
    if metadata['action_semantics']!='applied_absolute_normalized_targets':raise ValueError('Expected absolute applied targets')
    actions=np.load(prefix.with_suffix('.npz'))['actions'].astype(np.float32)
    if actions.shape!=(metadata['steps'],4,7):raise ValueError('Expected time x four poses x seven targets')
    if not np.isfinite(actions).all() or np.abs(actions).max()>1:raise ValueError('Invalid target values')
    if hashlib.sha256(actions.tobytes()).hexdigest()!=metadata['action_sha256']:raise ValueError('Action sequence checksum mismatch')
    cfg=EnvCfg();cfg.scene.num_envs=args.num_envs;cfg.action_mode='absolute'
    env=RecoverEnv(cfg);env.render_enabled=False;env.auto_reset=False;env.reset()
    if abs(metadata['control_dt']-env.step_dt)>1e-10:raise ValueError('Control period mismatch')
    if len(actions)>env.max_episode_length:raise ValueError('The sequence exceeds the current task time budget')
    fidelity=check_initial_body_poses(env)
    env.record_ids=torch.arange(args.num_envs,device=env.device)
    sequence=torch.from_numpy(actions).to(env.device)
    eligible=torch.ones(args.num_envs,dtype=torch.bool,device=env.device)
    succeeded=torch.zeros_like(eligible);first=torch.full((args.num_envs,),-1.,device=env.device)
    first_invalid=torch.full_like(first,-1.);max_hold=torch.zeros_like(first);traces=[]
    with torch.no_grad():
        for step in range(len(sequence)):
            env.step(sequence[step].repeat_interleave(args.num_envs//4,dim=0))
            first_invalid[env.last_step['invalid']&eligible]=(step+1)*env.step_dt
            eligible &= ~env.last_step['invalid']
            hit=env.last_step['success'] & eligible & ~succeeded
            first[hit]=(step+1)*env.step_dt;succeeded |= hit
            max_hold=torch.maximum(max_hold,torch.where(eligible,env.hold,0.))
            env.record['eligible']=eligible.detach().cpu().numpy().copy();traces.append(env.record)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'),**{k:np.stack([r[k] for r in traces]) for k in traces[0]})
    report=dict(status='completed',backend='physx',evaluation_mode='fixed_target_replay',
        source_hash=source_fingerprint(),evaluation_code_hash=task_implementation_fingerprint(),
        sequences=str(prefix),action_sha256=metadata['action_sha256'],num_envs=args.num_envs,
        per_pose_samples=args.num_envs//4,control_dt=env.step_dt,steps=len(actions),
        success_counts=succeeded.reshape(4,-1).sum(-1).cpu().tolist(),
        first_success_s=first.reshape(4,-1).cpu().tolist(),
        first_invalid_s=first_invalid.reshape(4,-1).cpu().tolist(),
        max_stable_s=max_hold.reshape(4,-1).cpu().tolist(),
        final_stable=(env.last_step['stable']&eligible).reshape(4,-1).cpu().tolist(),
        initial_fidelity=fidelity,placement=env.placement,
        joint_names=env.robot.joint_names,sensor_bodies=env.sensor.body_names,
        note='Evaluation-only absolute targets using the RL control interface; no desired-velocity feedforward or teacher training.')
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    env.close()
except BaseException:
    import traceback,os
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:
    app.close()
