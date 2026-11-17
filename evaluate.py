"""Independent deterministic PhysX evaluation with complete pre-reset trajectories."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--num-envs',type=int,default=64)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args(); app=AppLauncher(args).app
try:
    import numpy as np
    import torch
    from recover.env import RecoverEnv,EnvCfg
    from recover.policy import Policy
    from recover.contract import OBS_DIM
    from recover.model import source_fingerprint
    from recover.fidelity import check_initial_body_poses
    cfg=EnvCfg(); cfg.scene.num_envs=args.num_envs
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['source_hash']!=source_fingerprint(): raise ValueError('Source model changed')
    cfg.action_mode=checkpoint['args'].get('action_mode','absolute')
    cfg.delta_scale=checkpoint['args'].get('delta_scale',.12)
    env=RecoverEnv(cfg); env.render_enabled=False; env.auto_reset=False
    policy=Policy(width=checkpoint['args']['width']).to(env.device)
    policy.load_state_dict(checkpoint['model']); policy.eval()
    obs=env.reset()[0]['policy'].reshape(4,-1,OBS_DIM)
    initial_fidelity=check_initial_body_poses(env)
    env.record_ids=torch.arange(args.num_envs,device=env.device)
    traces=[]; succeeded=torch.zeros(args.num_envs,dtype=torch.bool,device=env.device)
    first=torch.full((args.num_envs,),-1.,device=env.device); max_hold=torch.zeros_like(first)
    eligible=torch.ones_like(succeeded); first_invalid=torch.full_like(first,-1.)
    with torch.no_grad():
        for t in range(env.max_episode_length):
            action=policy.act(obs).reshape(-1,7)
            obs=env.step(action)[0]['policy'].reshape(4,-1,OBS_DIM)
            first_invalid[env.last_step['invalid'] & eligible]=(t+1)*env.step_dt
            eligible &= ~env.last_step['invalid']
            hit=env.last_step['success'] & ~succeeded & eligible
            first[hit]=(t+1)*env.step_dt; succeeded |= env.last_step['success'] & eligible
            max_hold=torch.maximum(max_hold,torch.where(eligible,env.hold,0.))
            env.record['eligible']=eligible.detach().cpu().numpy().copy()
            traces.append(env.record)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'),**{k:np.stack([x[k] for x in traces]) for k in traces[0]})
    report=dict(checkpoint=str(args.checkpoint),iteration=checkpoint['iteration'],
                num_envs=args.num_envs,per_pose_samples=args.num_envs//4,
                success_counts=succeeded.reshape(4,-1).sum(-1).cpu().tolist(),
                first_success_s=first.reshape(4,-1).cpu().tolist(),
                first_invalid_s=first_invalid.reshape(4,-1).cpu().tolist(),
                max_stable_s=max_hold.reshape(4,-1).cpu().tolist(),
                final_stable=(env.last_step['stable']&eligible).reshape(4,-1).cpu().tolist(),
                joint_names=env.robot.joint_names,sensor_bodies=env.sensor.body_names,
                placement=env.placement,control_dt=env.step_dt,
                initial_fidelity=initial_fidelity,
                source_hash=checkpoint['source_hash'],status='completed',
                note='Deterministic replicas of four exact starts; not a random initial-state success rate.')
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True); env.close()
except BaseException:
    import traceback,sys,os
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
else:
    app.close()
