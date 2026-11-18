"""Replay an actually sampled target sequence on the unchanged source model."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
import torch
from recover.model import (workspace_root,source_fingerprint,task_implementation_fingerprint,
                           POSITION_JOINTS,POSITION_KP,POSITION_KD,FRONT_WHEELS,REAR_WHEELS)
from recover.poses import load_poses
from recover.contract import POSITION_LOW,POSITION_HIGH,CONTROL_DT,HOLD_S,posture
from evaluate_mujoco import render_video


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--witness',required=True,help='Prefix, .json or .npz of a recorded episode')
    parser.add_argument('--output',required=True)
    parser.add_argument('--physics-dt',type=float,default=.0005)
    parser.add_argument('--hold-final-seconds',type=float,default=2.)
    parser.add_argument('--video',action='store_true')
    args=parser.parse_args()
    code_hash=task_implementation_fingerprint()
    prefix=Path(args.witness);metadata=json.loads(prefix.with_suffix('.json').read_text())
    if metadata['source_hash']!=source_fingerprint():raise ValueError('The source model differs from the recording')
    if metadata['action_semantics']!='applied_absolute_normalized_targets':raise ValueError('Expected actual applied absolute targets')
    if abs(metadata['control_dt']-CONTROL_DT)>1e-10:raise ValueError('Control period mismatch')
    archive=np.load(prefix.with_suffix('.npz'));actions=np.asarray(archive['actions'],dtype=np.float32)
    if actions.shape!=(metadata['steps'],7) or not len(actions):raise ValueError('Invalid action sequence shape')
    if not np.isfinite(actions).all() or np.abs(actions).max()>1:raise ValueError('Invalid action targets')
    if hashlib.sha256(actions.tobytes()).hexdigest()!=metadata['action_sha256']:
        raise ValueError('Recorded action checksum mismatch')
    pose_id=metadata['pose_id'];pose_ids=[f'pose_{i:02d}' for i in range(1,5)]
    if pose_id not in pose_ids:raise ValueError('Unknown recorded initial pose')
    pose=pose_ids.index(pose_id);poses,addresses,placement=load_poses()
    if not np.allclose(archive['initial_root'][:7],poses[pose,:7],atol=1e-5,rtol=0):
        raise ValueError('Recorded root does not match the declared initial pose')
    expected=np.array([poses[pose,addresses[n]] for n in metadata['robot_joint_names']])
    if not np.allclose(archive['initial_joint'],expected,atol=1e-5,rtol=0):
        raise ValueError('Recorded joints do not match the declared initial pose')
    model=mujoco.MjModel.from_xml_path(str(workspace_root()/'robot_description/mjcf/scene.xml'))
    source_dt=float(model.opt.timestep)
    if not 0<args.physics_dt<=source_dt or abs(round(CONTROL_DT/args.physics_dt)*args.physics_dt-CONTROL_DT)>1e-10:
        parser.error('--physics-dt must divide 20 ms and be no larger than the source timestep')
    if args.hold_final_seconds<0 or abs(round(args.hold_final_seconds/CONTROL_DT)*CONTROL_DT-args.hold_final_seconds)>1e-10:
        parser.error('--hold-final-seconds must be a nonnegative multiple of 20 ms')
    model.opt.timestep=args.physics_dt;model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data=mujoco.MjData(model);data.qpos[:]=poses[pose];mujoco.mj_forward(model,data)
    jp=np.array([model.joint(n).qposadr[0] for n in POSITION_JOINTS])
    jd=np.array([model.joint(n).dofadr[0] for n in POSITION_JOINTS])
    pa=np.array([model.actuator(n).id for n in POSITION_JOINTS])
    wa=np.array([model.actuator(n).id for n in REAR_WHEELS]);wd=np.array([model.joint(n).dofadr[0] for n in REAR_WHEELS])
    wb=[model.body(n).id for n in FRONT_WHEELS+REAR_WHEELS]
    other=[i for i in range(1,model.nbody) if i not in wb]
    low=np.array(POSITION_LOW);high=np.array(POSITION_HIGH)
    kp=np.array([POSITION_KP[n] for n in POSITION_JOINTS]);kd=np.array([POSITION_KD[n] for n in POSITION_JOINTS])
    cap=model.actuator_forcerange[pa,1];decimation=round(CONTROL_DT/model.opt.timestep)
    velocity=np.zeros(6);contact_force=np.zeros(6);hold=0.;best=0.;first=-1.;first_invalid=-1.;eligible=True
    trace=[];count=len(actions)+round(args.hold_final_seconds/CONTROL_DT)
    for step in range(count):
        action=actions[min(step,len(actions)-1)]
        position=low+(action[:5]+1)*.5*(high-low);speed=25*action[5:]
        subforces=[];subground=[];subwheel=[];subeffort=[]
        for _ in range(decimation):
            data.ctrl[pa]=np.clip(kp*(position-data.qpos[jp])-kd*data.qvel[jd],-cap,cap)
            data.ctrl[wa]=speed;mujoco.mj_step(model,data)
            forces=np.zeros((model.nbody,3));ground=np.zeros_like(forces)
            for c in range(data.ncon):
                info=data.contact[c];mujoco.mj_contactForce(model,data,c,contact_force)
                force=info.frame.reshape(3,3).T@contact_force[:3]
                b1=model.geom_bodyid[info.geom1];b2=model.geom_bodyid[info.geom2]
                forces[b1]-=force;forces[b2]+=force
                if b1==0:ground[b2]+=force
                if b2==0:ground[b1]-=force
            subforces.append(forces);subground.append(ground)
            subwheel.append(data.qvel[wd].copy());subeffort.append(data.actuator_force.copy())
        mujoco.mj_forward(model,data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():raise RuntimeError('Nonfinite replay state')
        for name in ('mjWARN_BADQPOS','mjWARN_BADQVEL','mjWARN_BADQACC','mjWARN_BADCTRL'):
            if data.warning[getattr(mujoco.mjtWarning,name)].number:raise RuntimeError(f'Numerical failure: {name}')
        root=np.zeros((1,13),dtype=np.float32);root[0,:7]=data.qpos[:7]
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,model.body('base_link').id,velocity,0)
        root[0,7:10]=velocity[3:];root[0,10:13]=velocity[:3]
        history=np.stack(subforces)[None];contact=(history[:,:,wb,2]>.05).mean(1)>=.8
        otherforce=np.linalg.norm(history[:,:,other],axis=-1).max((1,2))
        wheel=data.qvel[wd].astype(np.float32)[None]
        up,four,stable,_=posture(torch.from_numpy(root),torch.from_numpy(contact),torch.from_numpy(otherforce),torch.from_numpy(wheel))
        invalid=bool(np.linalg.norm(root[0,:2])>3 or root[0,2]<-.1)
        if invalid and eligible:first_invalid=(step+1)*CONTROL_DT
        eligible=eligible and not invalid
        hold=hold+CONTROL_DT if bool(stable[0]) else 0.
        if eligible:best=max(best,hold)
        if eligible and hold>=HOLD_S and first<0:first=(step+1)*CONTROL_DT
        trace.append(dict(qpos=data.qpos.copy()[None],qvel=data.qvel.copy()[None],root=root,
            q=data.qpos[jp].astype(np.float32)[None],qd=data.qvel[jd].astype(np.float32)[None],
            wheel=wheel,action=action.copy()[None],ctrl=data.ctrl.copy()[None],effort=data.actuator_force.copy()[None],
            force=history,ground_force=np.stack(subground)[None],wheel_substeps=np.stack(subwheel)[None],
            actuator_substeps=np.stack(subeffort)[None],contacts=contact,other=otherforce,
            up=up.numpy(),four=four.numpy(),stable=stable.numpy(),hold=np.array([hold]),
            invalid=np.array([invalid]),eligible=np.array([eligible]),
            recorded_sequence=np.array([step<len(actions)])))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    arrays={k:np.stack([t[k] for t in trace]) for k in trace[0]}
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    report=dict(backend='original_mujoco',evaluation_mode='sampled_action_replay',pose_ids=[pose_id],
        witness=str(prefix),source_hash=source_fingerprint(),evaluation_code_hash=code_hash,
        policy_code_hash=metadata['code_hash'],mujoco_version=mujoco.__version__,integrator_choice='implicitfast',
        source_physics_dt=source_dt,physics_dt=args.physics_dt,physics_samples_per_control=decimation,
        control_dt=CONTROL_DT,recorded_steps=len(actions),hold_final_seconds=args.hold_final_seconds,
        success=[first>=0],success_during_recorded_sequence=[0<=first<=len(actions)*CONTROL_DT+1e-8],
        first_success_s=[first],first_invalid_s=[first_invalid],max_stable_s=[best],
        final_stable=[bool(trace[-1]['stable'][0]) and eligible],placement=[placement[pose]],
        body_names=[model.body(i).name for i in range(model.nbody)],status='completed',
        note='Exact recorded start and actually applied targets; any appended final-target hold is reported separately.')
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if args.video:render_video(model,arrays,output,pose_labels=[pose_id])


if __name__=='__main__':main()
