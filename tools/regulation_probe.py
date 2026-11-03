"""Finite real-physics checks for continuing STOP and RSL timeout semantics.

Scripted engineering actions only; no trajectory or state is PPO training data.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);app=AppLauncher(a).app
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from step_demo.config import PROJECT_ROOT,load_config
        from step_demo.runtime import load_training
        from step_demo.isaac_env import StepEnv
        t=load_training(PROJECT_ROOT/'configs/restart17_gate.json');t['num_envs']=16
        env=StepEnv(load_config(model_dir=a.model_dir),t,a.asset_dir,a.device)
        env.trace_ids=list(range(16));wrapped=RslRlVecEnvWrapper(env)
        assert not env.cfg.is_finite_horizon and not env.evaluation_mode
        assert (env.estimator.velocity==0).all()
        speeds=(torch.arange(16,device=env.device)%4).float()*.2
        serial=env.episode_serial.clone();steps=env.max_episode_length
        def braking(step):
            wheel=speeds/.03*max(0.,1-step*env.step_dt/.5)
            return torch.stack((torch.zeros_like(wheel),wheel/12.5-1),-1)
        def advance(actions):
            obs,reward,done,info=wrapped.step(actions)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            return done,info,env.last_trace
        longest_hold=torch.zeros(16,device=env.device)
        hold_breaks=[];previous_hold=torch.zeros_like(longest_hold)
        with (a.output/'holding_trace.jsonl').open('w') as f:
            for step in range(steps):
                done,info,row=advance(braking(step));f.write(json.dumps(row)+'\n')
                hold_now=torch.tensor(row['auxiliary_hold'],device=env.device)
                longest_hold=torch.maximum(longest_hold,hold_now)
                for i in torch.where((previous_hold>=2.5)&(hold_now==0))[0].tolist():
                    hold_breaks.append(dict(env=i,elapsed=row['elapsed'][i],
                        previous_hold_s=float(previous_hold[i]),support=row['supported_top'][i],
                        velocity=row['velocity'][i]))
                previous_hold=hold_now
                assert all(v==[0,0] for v in row['front_flight_count'])
                assert row['functional_completed']==(hold_now+1e-7>=env.task.acceptance.stop_hold_s).tolist()
                if step<steps-1:
                    assert not done.any() and not info['time_outs'].any()
                    assert torch.equal(env.episode_serial,serial)
                else:
                    assert done.all() and info['time_outs'].all()
                    assert not any(row['completed'])
                    # The accepted hold is 0.5+2s. Keep simulating to the
                    # 15s truncation even if a later real contact gap breaks
                    # the timer; record it rather than silently relaxing it.
                    assert (longest_hold>=env.task.acceptance.stop_hold_s+2.).all()
                    assert torch.equal(env.episode_serial,serial+1)
        hold=row['auxiliary_hold'];quality=[]
        # Settle the fresh reset before real motor-driven lifting.
        for step in range(200):advance(braking(step))
        with (a.output/'quality_trace.jsonl').open('w') as f:
            for step in range(100):
                hip=torch.where(torch.arange(16,device=env.device)%2==0,-1.,1.)
                done,info,row=advance(torch.stack((hip,torch.full_like(hip,-1.)),-1));f.write(json.dumps(row)+'\n')
                for i,code in enumerate(row['failure']):
                    if code==5:
                        assert done[i] and not info['time_outs'][i]
                        assert abs(row['reward'][i]+row['restart_running_cost'][i]+15.)<2e-5
                        quality.append(i)
                if quality:break
        assert quality
        wrapped.reset()
        for step in range(200):advance(braking(step))
        first_episode=env.episode_serial.clone();exits={}
        with (a.output/'exit_trace.jsonl').open('w') as f:
            for step in range(400):
                wheel=min(8.,step*env.step_dt*16.)
                actions=torch.tensor([0.,wheel/12.5-1.],device=env.device).expand(16,-1)
                done,info,row=advance(actions);f.write(json.dumps(row)+'\n')
                for i,code in enumerate(row['failure']):
                    if row['episode'][i]!=int(first_episode[i]) or i in exits:continue
                    if code:
                        assert code==6, (i,code)
                        assert done[i] and not info['time_outs'][i]
                        assert max(v[0] for v in row['wheel_max'][i])>=row['edge_x'][i]+env.task.terrain.platform_length_m
                        assert abs(row['reward'][i]+row['restart_running_cost'][i]+15.)<2e-5
                        exits[i]=row['elapsed'][i]
                if len(exits)==16:break
        assert len(exits)==16,exits
        result=dict(status='passed',probe_only=True,no_training_data=True,
            train_continues_after_first_stop=True,timeout_is_bootstrapped=True,
            real_failures_never_bootstrapped=True,final_hold_at_timeout_s=hold,
            longest_hold_s=longest_hold.tolist(),late_hold_breaks=hold_breaks,
            real_quality_failure_cases=len(quality),real_platform_exit_cases=len(exits),
            discounted_failure_cost=env.regulation_failure_cost,exit_elapsed_s=exits,
            captured_substeps=env.captured_substeps)
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');env.close()
    except BaseException:
        (a.output/'result.json').write_text(json.dumps(dict(status='failed',error=traceback.format_exc()),indent=2)+'\n')
        raise
    finally:app.close()


if __name__=='__main__':main()
