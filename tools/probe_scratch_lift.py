"""Physics-only bounded action grid; no network, teacher, or training dataset.

Tests whether the new rise reward responds to real motion under the unchanged
motor/model limits. Constant targets are diagnostic inputs, never PPO samples.
"""
import argparse,json,os,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
try:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from upright.feedback_task import FeedbackEnvCfg,ensure
    from upright.continuous_task import reward
    cfg=FeedbackEnvCfg();cfg.scene.num_envs=64;cfg.sim.device=args.device;cfg.seed=20261030
    cfg.from_scratch=True;cfg.decimation=10;cfg.sim.render_interval=10
    cfg.speed_limit=.75;cfg.yaw_limit=15.;cfg.frame_range=(0.,0.);cfg.hold_range=(4.,4.)
    cfg.wheel_slew=480.;cfg.frame_slew=3.;cfg.wheel_mode='common_diff';cfg.feedback_mode='pose_feedback_tight'
    cfg.tracking_objective='stability_first';cfg.observation_frame='balance_local';cfg.handoff_alignment='after'
    cfg.transition_contract='linear_heading';cfg.task_profile='full_range';cfg.training_mixture='off'
    cfg.capability_version='v25';cfg.mirror_training=False;cfg.adaptive_curriculum=False;cfg.evaluating=True
    cfg.evaluation_pattern='mixed';cfg.rise_speed_reference_mode='capture';cfg.yaw_feedback_design='fast2';cfg.yaw_feedback_gain=16.
    cfg.active_seconds=16.;cfg.episode_length_s=44.
    cfg.rewards.rise_progress=RewardTermCfg(func=reward,weight=4.,params={'kind':'rise_progress_reward'})
    env=ManagerBasedRLEnv(cfg=cfg);env.reset();progress=ensure(env)
    progress.set_command([0.,0.]);progress.request_mode(1)
    ids=torch.arange(64,device=env.device)
    hips=torch.tensor([-1.,-.6,-.2,0.,.3,.6,.9,1.2],device=env.device)[ids//8]
    wheel=torch.tensor([-25.,-16.,-8.,0.,8.,16.,25.,0.],device=env.device)[ids%8]
    active=torch.ones(64,device=env.device,dtype=torch.bool)
    gap=torch.zeros(64,device=env.device);height=gap.clone();credited=gap.clone();balance=active&False
    failures=torch.zeros(64,device=env.device,dtype=torch.long);events=[];previous_phase=torch.zeros_like(ids)
    wheel_peak=effort_peak=0.
    with torch.inference_mode():
        for tick in range(650):
            a=env.action_manager.get_term('moving');command=wheel.clone()
            # One column also probes a finite acceleration/reversal impulse.
            timed=ids%8==7;seconds=progress.phase_time
            command[timed]=torch.where(seconds[timed]<.35,25.,torch.where(seconds[timed]<.7,-25.,0.))
            action=torch.stack((2*(hips+1.)/2.2-1.,command/25.,torch.zeros_like(command)),-1)
            obs,r,term,timeout,extra=env.step(action);s,m,*_=env.moving_snapshot
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(r).all()
            assert m['wheel_target'].abs().max()<=25.0001
            assert m['effort'][:,-2:].abs().max()<=.38001
            wheel_peak=max(wheel_peak,float(m['wheel_target'].abs().max()));effort_peak=max(effort_peak,float(m['effort'][:,-2:].abs().max()))
            gap=torch.maximum(gap,torch.where(active,m['wheel_gap'].amin(-1),0.))
            height=torch.maximum(height,torch.where(active,s['front_z'],0.))
            credited+=active*m['rise_progress_reward']*env.step_dt
            balance|=active&m['entered_balance'];failures=torch.where(active&m['failure'],m['failure_code'],failures)
            changed=active&((m['phase']!=previous_phase)|m['failure'])
            for e in torch.where(changed)[0].tolist():
                events.append(dict(env=e,t=(tick+1)*env.step_dt,phase=int(m['phase'][e]),failure=int(m['failure_code'][e]),
                    joint_position=s['q'][e].tolist(),wheel_target=m['wheel_target'][e].tolist(),
                    wheel_actual=s['wheel'][e].tolist(),front_gap=m['wheel_gap'][e].tolist(),gravity=s['g'][e].tolist()))
            previous_phase=m['phase'];active&=~(term|timeout)
            if not active.any():break
    report=dict(engineering=True,policy=False,uses_network=False,uses_training_dataset=False,
        count=64,balance_entries=int(balance.sum()),max_front_gap=float(gap.max()),max_actual_front_height=float(height.max()),
        target_peak=wheel_peak,rear_effort_peak=effort_peak,
        cases=[dict(env=e,hip=float(hips[e]),common_wheel=float(wheel[e]),timed_impulse=e%8==7,
            max_gap=float(gap[e]),max_height=float(height[e]),progress_credit=float(credited[e]),
            balance=bool(balance[e]),failure=int(failures[e])) for e in range(64)],events=events)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','events')}))
    env.close()
except BaseException:
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
