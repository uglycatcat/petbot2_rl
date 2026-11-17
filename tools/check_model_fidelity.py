"""Compare original-MuJoCo and imported-PhysX link poses and inertial data."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',default='artifacts/engineering/model_fidelity.json')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args(); app=AppLauncher(args).app
try:
    import mujoco
    import numpy as np
    import torch
    from recover.env import RecoverEnv,EnvCfg,tensor
    from recover.model import source_dir,source_fingerprint
    from recover.poses import load_poses
    cfg=EnvCfg();cfg.scene.num_envs=64
    env=RecoverEnv(cfg);env.render_enabled=False;env.reset()
    robot=env.robot;names=robot.body_names
    ids=torch.arange(0,64,16,device=env.device)
    positions=tensor(robot.data.body_link_pos_w)[ids]-tensor(env.scene.env_origins)[ids,None,:]
    rotations=tensor(robot.data.body_link_quat_w)[ids][:,:,[3,0,1,2]]
    positions=positions.cpu().numpy();rotations=rotations.cpu().numpy()
    mj=mujoco.MjModel.from_xml_path(str(source_dir()/'mjcf/scene.xml'));d=mujoco.MjData(mj)
    poses,_,_=load_poses();body_ids=[mj.body(n).id for n in names]
    expected_pos=[];expected_quat=[]
    for qpos in poses:
        mujoco.mj_resetData(mj,d);d.qpos[:]=qpos;mujoco.mj_forward(mj,d)
        expected_pos.append(d.xpos[body_ids].copy());expected_quat.append(d.xquat[body_ids].copy())
    expected_pos=np.asarray(expected_pos);expected_quat=np.asarray(expected_quat)
    position_error=np.linalg.norm(positions-expected_pos,axis=-1)
    rotation_error=2*np.arccos(np.clip(np.abs((rotations*expected_quat).sum(-1)),0,1))
    view=robot.root_view
    masses=view.get_masses().numpy()[0]
    inertias=view.get_inertias().numpy()[0].reshape(-1,3,3)
    dofs={}
    for name in ('get_dof_armatures','get_dof_damping','get_dof_stiffnesses','get_dof_max_forces','get_dof_friction_coefficients'):
        if hasattr(view,name):
            value=getattr(view,name)().numpy()[0]
            dofs[name]=value.tolist()
    report=dict(source_hash=source_fingerprint(),body_names=names,joint_names=robot.joint_names,
                position_error_m=position_error.tolist(),rotation_error_rad=rotation_error.tolist(),
                physx_positions=positions.tolist(),mujoco_positions=expected_pos.tolist(),
                physx_rotations=rotations.tolist(),mujoco_rotations=expected_quat.tolist(),
                physx_mass=masses.tolist(),mujoco_mass=mj.body_mass[body_ids].tolist(),
                physx_inertia=inertias.tolist(),mujoco_principal_inertia=mj.body_inertia[body_ids].tolist(),
                dof_runtime=dofs,status='completed')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(position_max_m=float(position_error.max()),rotation_max_rad=float(rotation_error.max()),
                         mass_max_error=float(np.abs(masses-mj.body_mass[body_ids]).max()),dof_runtime=dofs)),flush=True)
    env.close()
except BaseException:
    import traceback
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:
    app.close()
