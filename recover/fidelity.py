"""Independent forward-kinematics oracle for the four recorded resets."""
import mujoco
import numpy as np
import torch
from .model import source_dir
from .poses import load_poses


def check_initial_body_poses(env):
    def tensor(value):
        return value if isinstance(value,torch.Tensor) else value.torch
    robot=env.robot
    ids=torch.arange(0,env.num_envs,env.num_envs//4,device=env.device)
    pos=(tensor(robot.data.body_link_pos_w)[ids]-tensor(env.scene.env_origins)[ids,None,:]).cpu().numpy()
    quat=tensor(robot.data.body_link_quat_w)[ids][:,:,[3,0,1,2]].cpu().numpy().astype(np.float64)
    quat/=np.linalg.norm(quat,axis=-1,keepdims=True)
    model=mujoco.MjModel.from_xml_path(str(source_dir()/'mjcf/scene.xml'))
    data=mujoco.MjData(model);poses,_,_=load_poses()
    body_ids=[model.body(n).id for n in robot.body_names]
    expected_pos=[];expected_quat=[]
    for pose in poses:
        mujoco.mj_resetData(model,data);data.qpos[:]=pose;mujoco.mj_forward(model,data)
        expected_pos.append(data.xpos[body_ids].copy());expected_quat.append(data.xquat[body_ids].copy())
    expected_pos=np.asarray(expected_pos);expected_quat=np.asarray(expected_quat)
    distance=np.linalg.norm(pos-expected_pos,axis=-1)
    angle=2*np.arccos(np.clip(np.abs((quat*expected_quat).sum(-1)),0,1))
    report=dict(body_names=robot.body_names,max_position_error_m=float(distance.max()),
                max_rotation_error_rad=float(angle.max()),position_errors_m=distance.tolist(),
                rotation_errors_rad=angle.tolist(),internal_quaternion='wxyz',simulator_quaternion='xyzw')
    # Large scene origins (hundreds of metres) incur float32 translation rounding.
    if distance.max()>1e-4 or angle.max()>1e-3:
        raise RuntimeError(f'Physical reset differs from original MuJoCo: {report}')
    return report
