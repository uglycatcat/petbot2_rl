"""Container GPU checks for exact resets, standing success and finite contact dynamics."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--num-envs',type=int,default=64)
parser.add_argument('--steps',type=int,default=400)
parser.add_argument('--output',default='artifacts/engineering/physics')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
try:
    import numpy as np
    import torch
    from recover.env import RecoverEnv,EnvCfg,tensor
    from recover.contract import POSITION_LOW,POSITION_HIGH
    cfg=EnvCfg(); cfg.scene.num_envs=args.num_envs
    env=RecoverEnv(cfg); env.render_enabled=False
    obs,_=env.reset()
    materials=torch.from_numpy(env.robot.root_view.get_material_properties().numpy())
    assert materials.shape[1:]==(10,3),materials.shape
    actual_friction=materials[:,:,1]
    assert torch.allclose(actual_friction.sort(-1).values,
        torch.tensor([.1]*8+[1.5]*2).expand_as(actual_friction),atol=1e-6),actual_friction[0]
    root,q,qd,wheel,contact,other=env.state()
    root_error=float((root-env.initial_root[env.pose_ids]).abs().max())
    allq_error=float((tensor(env.robot.data.joint_pos)-env.initial_joint[env.pose_ids]).abs().max())
    assert root_error<1e-5 and allq_error<1e-6,(root_error,allq_error)
    assert float(tensor(env.robot.data.joint_vel).abs().max())==0
    env.auto_reset=False; env.standing_probe=True
    env.reset()
    home=torch.zeros(args.num_envs,7,device=env.device)
    home[:,:5]=2*(0-env.low)/(env.high-env.low)-1
    for _ in range(150): env.step(home)
    s=env.last_step
    standing={k:v.detach().cpu().tolist() for k,v in s.items() if k in ('up','four','stable','hold','other','contacts')}
    assert s['stable'].all(),{k:v[:4] for k,v in standing.items()}
    assert (s['hold']>=.5).all(),standing['hold'][:4]
    env.standing_probe=False; env.reset()
    env.record_ids=torch.arange(0,args.num_envs,args.num_envs//4,device=env.device)
    traces=[]; peaks=[]
    for step in range(args.steps):
        # Deliberate temporally held random targets exercise joint ranges and wheel reversals.
        if step%10==0: action=2*torch.rand_like(home)-1
        obs,reward,_,_,_=env.step(action)
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        assert torch.isfinite(tensor(env.robot.data.root_state_w)).all()
        traces.append(env.record)
        peaks.append(float(tensor(env.robot.data.applied_torque).abs().max()))
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(reset_root_max_error=root_error,reset_joint_max_error=allq_error,
                placement=env.placement,standing=standing,joint_names=env.robot.joint_names,
                sensor_bodies=env.sensor.body_names,random_steps=args.steps,
                random_torque_peak=max(peaks),num_envs=args.num_envs,
                runtime_shape_materials=materials[0].tolist(),
                device=env.device,success_contract='four_wheels_upright_quiet_0.5s',status='passed')
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'),**{k:np.stack([t[k] for t in traces]) for k in traces[0]})
    print(json.dumps({k:v for k,v in report.items() if k!='standing'}),flush=True)
    env.close()
except BaseException:
    import traceback,sys,os
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
else:
    app.close()
