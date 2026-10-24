"""Fixed-v10 diagnostic of bounded fast forward correction during balance only."""
import argparse
import hashlib
import json
import pathlib
import sys

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--common-kp',type=float,choices=[0.,10.,30.],default=0.)
options,remaining=p.parse_known_args();assert '--evaluate' in remaining
sys.argv=[sys.argv[0],*remaining];sys.path.insert(0,str(pathlib.Path.cwd()))


def install_common_feedback(env,folder):
    import torch
    from upright.continuous_task import ensure
    from upright.cycle_task import tensor
    from upright.ground_motion import wheel_heading
    a=env.action_manager.get_term('moving');original=a.apply_actions
    assert env.cfg.yaw_feedback_design=='fast20' and env.cfg.decimation==30
    correction=torch.zeros(env.num_envs,device=env.device)
    def apply():
        index=a.inner_index
        original()
        if options.common_kp==0:return
        p=ensure(env);mask=(p.phase==2)&(p.phase_time>=1.2)
        correction[~mask]=0.
        if index%10==0:
            d=a.robot.data;pose=tensor(d.root_link_pose_w);velocity=tensor(d.root_com_vel_w)
            forward=wheel_heading(pose[:,3:7],velocity)[0]
            measured=(velocity[:,:2]*forward).sum(-1)
            desired=(options.common_kp*(p.command[:,0]-measured)).clamp(-1.,1.)
            correction[mask]+=(desired-correction).clamp(-.24,.24)[mask]
            common=a.inner_policy.mean(-1)+correction
            half=(a.speed[:,1]-a.speed[:,0])*.5
            anchor=a.inner_anchor;limit=env.cfg.wheel_slew*env.step_dt
            lower=torch.stack((-a.max_speed+half.abs(),anchor[:,0]-limit+half,anchor[:,1]-limit-half),-1).amax(-1)
            upper=torch.stack((a.max_speed-half.abs(),anchor[:,0]+limit+half,anchor[:,1]+limit-half),-1).amin(-1)
            common=torch.maximum(torch.minimum(common,upper),lower)
            out=torch.stack((common-half,common+half),-1)
            a.speed[mask]=out[mask];a.inner_speed[mask]=out[mask]
            a.applied[:,1:]=a.speed/a.max_speed
            a.robot.set_joint_velocity_target(a.speed,joint_ids=a.wheel_ids)
    a.apply_actions=apply
    (folder/'common_probe.json').write_text(json.dumps(dict(diagnostic_only=True,kp=options.common_kp,
        correction_limit=1.,correction_rate=12.,period=.02,active='phase2 clock>=1.2',
        source_hash=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2))


source=pathlib.Path('train_cycle.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,'Fixed evaluator changed'
    source=source.replace(old,new)
replace('obs=wrapped.get_observations();n=args.num_envs','install_common_feedback(env,folder);obs=wrapped.get_observations();n=args.num_envs')
replace("folder/'evaluation.json'","folder/'diagnostic_evaluation.json'")
replace("folder/'trace.json'","folder/'diagnostic_trace.json'")
exec(compile(source,'train_cycle.py','exec'),globals())
