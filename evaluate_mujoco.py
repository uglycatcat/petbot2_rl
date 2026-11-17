"""Validate saved policies on the unchanged MJCF with explicit source-limited PD."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
import torch

from recover.model import (workspace_root,POSITION_JOINTS,POSITION_KP,POSITION_KD,
                           FRONT_WHEELS,REAR_WHEELS,source_fingerprint,task_implementation_fingerprint)
from recover.poses import load_poses
from recover.policy import Policy
from recover.contract import observation,posture,POSITION_LOW,POSITION_HIGH,CONTROL_DT,HOLD_S


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=float,default=16.)
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--integrator',choices=['source','implicitfast'],default='implicitfast')
    parser.add_argument('--physics-dt',type=float,
                        help='Optional integration-resolution check; source XML and control period stay unchanged')
    args=parser.parse_args()
    evaluation_code_hash=task_implementation_fingerprint()
    if bool(args.checkpoint)==args.baseline: parser.error('Select checkpoint OR baseline')
    root=workspace_root(); model=mujoco.MjModel.from_xml_path(str(root/'robot_description/mjcf/scene.xml'))
    source_dt=float(model.opt.timestep)
    if args.integrator=='implicitfast': model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if args.physics_dt is not None:
        if not 0 < args.physics_dt <= source_dt:
            parser.error('--physics-dt must be positive and no larger than the source timestep')
        if abs(round(CONTROL_DT/args.physics_dt)*args.physics_dt-CONTROL_DT)>1e-10:
            parser.error('--physics-dt must divide the control period exactly')
        model.opt.timestep=args.physics_dt
    states=[mujoco.MjData(model) for _ in range(4)]
    poses,addresses,placement=load_poses()
    for i,data in enumerate(states): data.qpos[:]=poses[i]; mujoco.mj_forward(model,data)
    jp=np.array([model.joint(n).qposadr[0] for n in POSITION_JOINTS])
    jd=np.array([model.joint(n).dofadr[0] for n in POSITION_JOINTS])
    wa=np.array([model.actuator(n).id for n in REAR_WHEELS]); pa=np.array([model.actuator(n).id for n in POSITION_JOINTS])
    wd=np.array([model.joint(n).dofadr[0] for n in REAR_WHEELS])
    wb=[model.body(n).id for n in FRONT_WHEELS+REAR_WHEELS]
    other=[i for i in range(1,model.nbody) if i not in wb]
    low=np.array(POSITION_LOW); high=np.array(POSITION_HIGH)
    kp=np.array([POSITION_KP[n] for n in POSITION_JOINTS]); kd=np.array([POSITION_KD[n] for n in POSITION_JOINTS])
    cap=model.actuator_forcerange[pa,1]
    decimation=round(CONTROL_DT/model.opt.timestep)
    controls=[]; checkpoint=None
    if args.baseline:
        sys.path.insert(0,str(root/'robot_controller_demo/pose_recover_demo'))
        from control import RecoveryController,blend
        from attitude import read_imu
        controls=[RecoveryController(model) for _ in states]
        for control,data in zip(controls,states): control.request_stand(data,float(read_imu(model,data)[1][1]),True)
    else:
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if source_fingerprint()!=checkpoint['source_hash']: raise ValueError('Source model changed')
        policy=Policy(width=checkpoint['args']['width'],
                      action_mode=checkpoint['args'].get('action_mode','absolute'),
                      noise_rho=checkpoint['args'].get('noise_rho',0.))
        policy.load_state_dict(checkpoint['model']); policy.eval()
    last=np.zeros((4,7),dtype=np.float32); last[:,:5]=2*(poses[:,jp]-low)/(high-low)-1
    contact=np.zeros((4,4),dtype=bool); hold=np.zeros(4); best=np.zeros(4); first=np.full(4,-1.)
    eligible=np.ones(4,dtype=bool); first_invalid=np.full(4,-1.)
    trace=[]; dt=model.opt.timestep; count=round(args.seconds/CONTROL_DT)
    vel=np.zeros(6)
    def state_arrays():
        r=np.zeros((4,13),dtype=np.float32)
        for i,data in enumerate(states):
            r[i,:7]=data.qpos[:7]
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,model.body('base_link').id,vel,0)
            r[i,7:10]=vel[3:]; r[i,10:13]=vel[:3]
        return r,np.stack([d.qpos[jp] for d in states]).astype(np.float32),np.stack([d.qvel[jd] for d in states]).astype(np.float32),np.stack([d.qvel[wd] for d in states]).astype(np.float32)
    with torch.no_grad():
        for step in range(count):
            r,q,qd,wheel=state_arrays()
            obs=observation(torch.from_numpy(r),torch.from_numpy(q),torch.from_numpy(qd),torch.from_numpy(wheel),
                            torch.from_numpy(last),torch.from_numpy(contact),torch.full((4,),1-step/count))
            requested=np.zeros_like(last)
            if not args.baseline:
                requested=policy.act(obs[:,None,:])[:,0].numpy()
                if checkpoint['args'].get('action_mode','absolute')=='delta':
                    last=np.clip(last+checkpoint['args'].get('delta_scale',.12)*requested,-1,1)
                else: last=requested.copy()
            pos=low+(last[:,:5]+1)*.5*(high-low); speed=last[:,5:]*25
            subforces=[];subground=[];subwheel=[];subeffort=[]
            baseline_pos=np.zeros((4,5)); baseline_speed=np.zeros((4,2))
            for sub in range(decimation):
                forces=np.zeros((4,model.nbody,3))
                groundforces=np.zeros_like(forces)
                for i,data in enumerate(states):
                    if args.baseline:
                        control=controls[i]
                        if control.standing:
                            p,_,w=control._stand_reference(control._stand_elapsed)
                        else:
                            p,_=blend(control.q_start,control.home,min(control.elapsed+dt,control.DURATION),control.DURATION)
                            w=0.
                        baseline_pos[i]=p; baseline_speed[i]=w
                        control.step(data,dt)
                    else:
                        data.ctrl[pa]=np.clip(kp*(pos[i]-data.qpos[jp])-kd*data.qvel[jd],-cap,cap)
                        data.ctrl[wa]=speed[i]
                    mujoco.mj_step(model,data)
                    if not np.isfinite(data.qpos).all(): raise RuntimeError('Nonfinite MuJoCo state')
                    for c in range(data.ncon):
                        info=data.contact[c]; mujoco.mj_contactForce(model,data,c,vel)
                        force=info.frame.reshape(3,3).T@vel[:3]
                        b1=model.geom_bodyid[info.geom1]; b2=model.geom_bodyid[info.geom2]
                        forces[i,b1]-=force; forces[i,b2]+=force
                        if b1==0: groundforces[i,b2]+=force
                        if b2==0: groundforces[i,b1]-=force
                subforces.append(forces)
                subground.append(groundforces)
                subwheel.append(np.stack([d.qvel[wd].copy() for d in states]))
                subeffort.append(np.stack([d.actuator_force.copy() for d in states]))
            history=np.stack(subforces,axis=1)
            contact=(history[:,:,wb,2]>.05).mean(1)>=.8
            otherforce=np.linalg.norm(history[:,:,other,:],axis=-1).max((1,2))
            for data in states: mujoco.mj_forward(model,data)
            r,q,qd,wheel=state_arrays()
            up,four,stable,potential=posture(torch.from_numpy(r),torch.from_numpy(contact),torch.from_numpy(otherforce),torch.from_numpy(wheel))
            invalid=(np.linalg.norm(r[:,:2],axis=-1)>3.) | (r[:,2]<-.1)
            newly_invalid=invalid & eligible
            first_invalid[newly_invalid]=(step+1)*CONTROL_DT; eligible &= ~invalid
            for data in states:
                if not np.isfinite(data.qvel).all(): raise RuntimeError('Nonfinite MuJoCo velocity')
                for name in ('mjWARN_BADQPOS','mjWARN_BADQVEL','mjWARN_BADQACC','mjWARN_BADCTRL'):
                    if data.warning[getattr(mujoco.mjtWarning,name)].number:
                        raise RuntimeError(f'MuJoCo numerical failure: {name}')
            if args.baseline:
                last[:,:5]=2*(baseline_pos-low)/(high-low)-1; last[:,5:]=baseline_speed/25
            hold=np.where(stable.numpy(),hold+CONTROL_DT,0.)
            best=np.maximum(best,np.where(eligible,hold,0.))
            hit=(hold>=HOLD_S)&(first<0)&eligible; first[hit]=(step+1)*CONTROL_DT
            trace.append(dict(qpos=np.stack([d.qpos.copy() for d in states]),qvel=np.stack([d.qvel.copy() for d in states]),
                root=r,q=q,qd=qd,wheel=wheel,action=last.copy(),ctrl=np.stack([d.ctrl.copy() for d in states]),
                policy_action=requested.copy(),
                effort=np.stack([d.actuator_force.copy() for d in states]),
                eligible=eligible.copy(),invalid=invalid.copy(),
                force=history,contacts=contact.copy(),other=otherforce,up=up.numpy(),four=four.numpy(),
                ground_force=np.stack(subground,axis=1),
                wheel_substeps=np.stack(subwheel,axis=1),
                actuator_substeps=np.stack(subeffort,axis=1),
                stable=stable.numpy(),hold=hold.copy()))
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    arrays={k:np.stack([t[k] for t in trace]) for k in trace[0]}
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    report=dict(backend='original_mujoco',mujoco_version=mujoco.__version__,
                integrator=int(model.opt.integrator),integrator_choice=args.integrator,source_hash=source_fingerprint(),
                checkpoint=args.checkpoint,baseline=args.baseline,success=(first>=0).tolist(),
                evaluation_code_hash=evaluation_code_hash,
                policy_code_hash=checkpoint.get('code_hash') if checkpoint else None,
                first_success_s=first.tolist(),first_invalid_s=first_invalid.tolist(),
                max_stable_s=best.tolist(),final_stable=(trace[-1]['stable']&eligible).tolist(),
                control_dt=CONTROL_DT,placement=placement,status='completed',
                source_physics_dt=source_dt,physics_dt=float(model.opt.timestep),
                physics_samples_per_control=decimation,
                body_names=[model.body(i).name for i in range(model.nbody)],
                note='Four exact recorded starts; deterministic validation, not broad initial-state coverage.')
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if args.video: render_video(model,arrays,output)


def render_video(model,arrays,output,pose_labels=None):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    data=mujoco.MjData(model); camera=mujoco.MjvCamera()
    camera.distance=.7; camera.azimuth=135; camera.elevation=-22
    renderer=mujoco.Renderer(model,height=480,width=640)
    count=arrays['qpos'].shape[1];cols=min(count,2);rows=(count+cols-1)//cols
    pose_labels=pose_labels or [f'pose_{i+1:02d}' for i in range(count)]
    writer=imageio.get_writer(str(output.with_suffix('.mp4')),fps=25,codec='libx264',quality=7)
    try:
        for t in range(0,len(arrays['qpos']),2):
            canvas=Image.new('RGB',(640*cols,480*rows))
            for i in range(count):
                data.qpos[:]=arrays['qpos'][t,i]; mujoco.mj_forward(model,data)
                camera.lookat[:]=[data.qpos[0],data.qpos[1],.09]
                renderer.update_scene(data,camera=camera)
                frame=Image.fromarray(renderer.render().copy()); draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,640,30),fill=(0,0,0))
                draw.text((8,8),f'{pose_labels[i]}  t={(t+1)*CONTROL_DT:.2f}s  stable_hold={arrays["hold"][t,i]:.2f}s',fill='white')
                if 'recorded_sequence' in arrays and not arrays['recorded_sequence'][t,i]:
                    draw.rectangle((0,30,640,55),fill=(90,60,0))
                    draw.text((8,37),'Post-sequence hold: last recorded target',fill='white')
                if 'eligible' in arrays and not arrays['eligible'][t,i]:
                    draw.rectangle((0,450,640,480),fill=(110,0,0))
                    draw.text((8,458),'Outside task bounds: later states are not eligible for success',fill='white')
                canvas.paste(frame,((i%cols)*640,(i//cols)*480))
            writer.append_data(np.asarray(canvas))
    finally: writer.close(); renderer.close()


if __name__=='__main__': main()
