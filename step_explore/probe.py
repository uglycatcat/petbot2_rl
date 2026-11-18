"""Targeted real-PhysX checks; no trained weights and no demonstrations fed into PPO."""
import json
from pathlib import Path
import numpy as np
import torch

def run_probe(env,wrapped,output):
    checks={'step_height':env.cfg.height,'control_dt':env.step_dt}
    obs,_=wrapped.reset()
    assert torch.isfinite(obs['policy']).all()
    zero=torch.zeros(env.num_envs,5,device=env.device)
    for _ in range(max(1,round(1.5/env.step_dt))):wrapped.step(zero)
    root=env.robot.data.root_pos_w-env.scene.env_origins
    checks['neutral_root_height_range']=[float(root[:,2].min()),float(root[:,2].max())]
    checks['neutral_joint_abs_max']=float(env.robot.data.joint_pos.abs().max())
    checks['source_mass']=env.meta['mass']
    masses=env.robot.root_physx_view.get_masses()
    checks['physx_mass']=float(masses[0].sum())
    assert abs(checks['physx_mass']-checks['source_mass'])<1e-5
    assert root[:,2].min()>.04 and root[:,2].max()<.15
    assert not env.success.any()
    assert 'left_ear' not in env.robot.joint_names and 'right_ear' not in env.robot.joint_names
    print('NEUTRAL_CHECK',json.dumps(checks),flush=True)
    # Same default pose translated onto platform: the task must recognize physical four-wheel support.
    wrapped.reset()
    state=env.robot.data.default_root_state.clone()
    state[:,:3]+=env.scene.env_origins
    state[:,0]+=env.cfg.edge+.35
    state[:,2]+=env.cfg.height
    env.robot.write_root_pose_to_sim(state[:,:7]);env.robot.write_root_velocity_to_sim(state[:,7:])
    observed=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    for _ in range(max(1,round(.5/env.step_dt))):
        wrapped.step(zero)
        observed |= env.last_transition['success']
    checks['positive_successes']=int(observed.sum())
    assert observed.all(),checks
    # Airborne directly above the platform cannot pass just because x exceeds the edge.
    wrapped.reset();state[:,2]+=.4
    env.robot.write_root_pose_to_sim(state[:,:7]);env.robot.write_root_velocity_to_sim(state[:,7:])
    wrapped.step(zero)
    assert not env.last_transition['success'].any()
    checks['airborne_negative']=True
    # Independent angle and wheel-speed setpoints, with both directions exercised.
    wrapped.reset()
    actions=torch.zeros_like(zero);actions[:,0]=.25;actions[:,1]=.2;actions[:,2]=-.2
    actions[:,3]=.2;actions[:,4]=-.2
    state=env.robot.data.default_root_state.clone();state[:,:3]+=env.scene.env_origins
    state[:,2]+=.6
    env.robot.write_root_pose_to_sim(state[:,:7]);env.robot.write_root_velocity_to_sim(state[:,7:])
    for _ in range(max(1,round(.2/env.step_dt))):wrapped.step(actions)
    checks['independent_joint_targets']=env.targets[0].cpu().tolist()
    checks['actual_joint_positions']=env.robot.data.joint_pos[0].cpu().tolist()
    checks['actual_joint_velocities']=env.robot.data.joint_vel[0].cpu().tolist()
    assert env.targets[0,1]>0 and env.targets[0,2]<0 and env.targets[0,3]>0 and env.targets[0,4]<0
    checks['rear_velocity_error_max']=float((env.robot.data.joint_vel[:,env.rear_ids]-env.targets[:,3:]).abs().max())
    checks['joint_names']=env.robot.joint_names
    checks['rear_ids']=env.rear_ids
    checks['drive_damping']=env.robot.root_physx_view.get_dof_dampings()[0].tolist()
    checks['drive_max_force']=env.robot.root_physx_view.get_dof_max_forces()[0].tolist()
    assert checks['rear_velocity_error_max']<.5,checks
    assert torch.isfinite(env.robot.data.root_state_w).all()
    checks['passed']=True
    Path(output,'probe.json').write_text(json.dumps(checks,indent=2))
    print('PHYSICS_PROBE',json.dumps(checks),flush=True)
