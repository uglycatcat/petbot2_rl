"""Fixed-v9 diagnostic: faster yaw feedback only after stable two-wheel entry.

No checkpoint/task-contract rewrite. Diagnostic results never enter batch scores.
"""
import argparse
import hashlib
import json
import pathlib
import sys

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--feedback-probe',choices=['baseline','instant20','filtered20'],default='baseline')
p.add_argument('--feedback-kp',type=float,default=2.)
p.add_argument('--feedback-rate',type=float,default=4.)
options,remaining=p.parse_known_args();assert '--evaluate' in remaining
sys.argv=[sys.argv[0],*remaining];sys.path.insert(0,str(pathlib.Path.cwd()))


def install_fast_feedback(env,folder):
    import torch
    from upright.continuous_task import ensure
    from upright.cycle_task import tensor
    from upright.feedback import YawFeedback
    from upright.continuous import yaw_feedforward
    from upright.ground_motion import wheel_heading
    a=env.action_manager.get_term('moving')
    process_old=a.process_actions;apply_old=a.apply_actions
    fast=YawFeedback(env.num_envs,env.device,.02,a.max_speed,env.cfg.wheel_slew)
    probe=dict(index=0,mask=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool),
               speed=a.speed.clone(),policy=a.speed.clone(),filtered=torch.zeros(env.num_envs,device=env.device))
    assert abs(env.step_dt-.06)<1e-9 and abs(env.physics_dt-.002)<1e-9
    com=tensor(a.robot.data.body_com_pose_b)[:,0,:3]
    root_check=dict(root_body=a.robot.body_names[0],max_root_com_offset=float(com.norm(dim=-1).max()),
                    max_velocity_difference=0.)

    def process(actions):
        previous=a.speed.clone();integral=a.yaw_feedback.integral.clone()
        process_old(actions)
        progress=ensure(env)
        mask=(progress.phase==2)&(progress.phase_time>=1.2)
        probe['index']=0;probe['mask']=mask
        probe['policy']=a.speed.clone();probe['speed']=previous
        fast.integral.copy_(integral)
        # Initialize the shorter low-pass from the latest ground-rate sample.
        root=tensor(a.robot.data.root_state_w)
        root_check['max_velocity_difference']=max(root_check['max_velocity_difference'],
            float((tensor(a.robot.data.root_com_vel_w)-tensor(a.robot.data.root_link_vel_w)).abs().max()))
        rate=wheel_heading(root[:,3:7],root[:,7:13])[2]
        probe['filtered']=torch.where(mask,probe['filtered'],rate)
        if options.feedback_probe!='baseline':
            a.yaw_feedback.integral[mask]=integral[mask]

    def apply():
        i=probe['index'];mask=probe['mask']
        if options.feedback_probe!='baseline' and i%10==0:
            progress=ensure(env)
            root=tensor(a.robot.data.root_state_w)
            rate=wheel_heading(root[:,3:7],root[:,7:13])[2]
            probe['filtered']+=.5*(rate-probe['filtered'])
            measured=progress.motion.clone()
            measured[:,1]=rate if options.feedback_probe=='instant20' else probe['filtered']
            candidate=fast.process(probe['speed'],probe['policy'],progress.phase,progress.phase_time,
                                   progress.command,measured,a.raw_actions[:,2])
            desired=(yaw_feedforward(progress.command[:,1])+options.feedback_kp*(progress.command[:,1]-measured[:,1])
                     +fast.integral+.25*a.raw_actions[:,2]).clamp(-2.,2.)
            old=(probe['speed'][:,1]-probe['speed'][:,0])*.5
            fast.target=old+(desired-old).clamp(-options.feedback_rate*.02,options.feedback_rate*.02)
            common=probe['policy'].mean(-1)
            margin=(a.max_speed-common.abs()).clamp_min(0)
            half=torch.maximum(torch.minimum(fast.target,margin),-margin)
            candidate=torch.stack((common-half,common+half),-1)
            probe['speed'][mask]=candidate[mask]
            a.speed[mask]=candidate[mask]
            a.applied[:,1:]=a.speed/a.max_speed
            a.yaw_feedback.integral[mask]=fast.integral[mask]
            a.yaw_feedback.target[mask]=fast.target[mask]
        apply_old();probe['index']+=1

    a.process_actions=process;a.apply_actions=apply
    def save():
        (folder/'root_reference_check.json').write_text(json.dumps(root_check,indent=2))
    (folder/'feedback_probe.json').write_text(json.dumps(dict(diagnostic_only=True,mode=options.feedback_probe,kp=options.feedback_kp,difference_rate=options.feedback_rate,
        intervention='phase2 clock>=1.2; yaw PI at20ms; learned common/frame unchanged; physics and policy cadence unchanged',
        source_hash=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2))
    return save


source=pathlib.Path('train_cycle.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,'Evaluator changed'
    source=source.replace(old,new)
replace('obs=wrapped.get_observations();n=args.num_envs','save_probe=install_fast_feedback(env,folder);obs=wrapped.get_observations();n=args.num_envs')
replace('        result=dict(max_wheel_axis_alignment_error=', '        save_probe()\n        result=dict(max_wheel_axis_alignment_error=')
replace("folder/'evaluation.json'","folder/'diagnostic_evaluation.json'")
replace("folder/'trace.json'","folder/'diagnostic_trace.json'")
exec(compile(source,'train_cycle.py','exec'),globals())
