"""Fresh actor approach, real PPO, and train/eval quality semantics in PhysX.

No weights or trajectories are loaded. Motor-only probes do not enter PPO.
This validates exploration and engineering, never learned full-task capability.
"""
import argparse
import json
import math
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
        from step_demo.ppo import StepRunner,runner_config,state_digest,distribution_metrics
        t=load_training(PROJECT_ROOT/'configs/restart21_gate.json');t['num_envs']=16
        env=StepEnv(load_config(model_dir=a.model_dir),t,a.asset_dir,a.device)
        env.trace_ids=list(range(16));wrapped=RslRlVecEnvWrapper(env)
        runner=StepRunner(wrapped,runner_config(t),{'training':t},device=a.device)
        assert not runner.alg.optimizer.state
        assert all(x.requires_grad for x in runner.alg.actor.parameters())
        initial=state_digest(runner.alg.actor.state_dict());runner.alg.actor.eval()
        obs=wrapped.get_observations();distance=[0.]*16;done_ids=set()
        assert obs['policy'].shape==(16,26) and (env.packet_count==1).all()
        with (a.output/'initial_mean_trace.jsonl').open('w') as f,torch.inference_mode():
            for _ in range(150):
                obs,_,_,_=wrapped.step(runner.alg.actor(obs));row=env.last_trace;f.write(json.dumps(row)+'\n')
                for k in range(16):
                    if k in done_ids:continue
                    distance[k]=max(distance[k],row['com'][k][0])
                    if row['failure'][k] or row['elapsed'][k]+1e-5>=15:done_ids.add(k)
                if len(done_ids)==16:break
        assert min(distance)>.05,('Fresh mean failed to approach',distance)
        assert state_digest(runner.alg.actor.state_dict())==initial
        runner.alg.actor.train()
        with torch.inference_mode():
            wrapped.reset();obs=wrapped.get_observations()
        outside=0;density_error=0.;reward_error=0.;prior_landed=[[False,False] for _ in range(16)]
        prior_episode=[None]*16
        with (a.output/'ppo_trace.jsonl').open('w') as f,torch.inference_mode():
            for step in range(t['num_steps_per_env']):
                old_executed=env.executed.clone()
                raw=runner.alg.act(obs);original=raw.clone();lp=runner.alg.actor.get_output_log_prob(raw).clone()
                obs,reward,done,info=wrapped.step(raw);row=env.last_trace;f.write(json.dumps(row)+'\n')
                torch.testing.assert_close(raw,original,rtol=0,atol=0)
                torch.testing.assert_close(torch.tensor(row['action'],device=env.device),raw,rtol=0,atol=0)
                torch.testing.assert_close(torch.tensor(row['requested_action'],device=env.device),raw.clamp(-1,1),rtol=0,atol=0)
                runner.alg.process_env_step(obs,reward,done,info)
                torch.testing.assert_close(runner.alg.storage.actions[step],raw,rtol=0,atol=0)
                density_error=max(density_error,float((runner.alg.storage.actions_log_prob[step].squeeze(-1)-lp).abs().max()))
                outside+=int((raw.abs()>1).sum())
                for k in range(16):
                    if prior_episode[k]!=row['episode'][k]:prior_landed[k]=[False,False]
                    failed=row['failure'][k]>0 or (row['elapsed'][k]+1e-5>=15. and not row['completed'][k])
                    landing=sum(x and not y for x,y in zip(row['front_landed'][k],prior_landed[k]))
                    progress=row['shaping_reward'][k]+2*landing*(not failed)
                    physical=sum(row[n][k] for n in ['landing_cost','motion_cost','relift_cost','touch_impact_cost','touch_motion_cost'])
                    delta=sum((row['executed'][k][j]-float(old_executed[k,j]))**2 for j in range(2))
                    line_cost=env.step_dt*(2*row['com'][k][1]**2+row['yaw'][k]**2+all(row['front_landed'][k])*sum(row['joint_position'][k][h]**2 for h in env.hips)/2)
                    expected=progress-physical-.02*delta-line_cost-row['restart_cost_components'][k][6]+(20.-.1*row['elapsed'][k])*row['completed'][k]-5*failed
                    reward_error=max(reward_error,abs(expected-row['reward'][k]))
                    if failed:assert not bool(info.get('time_outs',torch.zeros(16,device=env.device))[k])
                    prior_episode[k]=row['episode'][k];prior_landed[k]=row['front_landed'][k]
            runner.alg.compute_returns(obs)
        assert outside>0 and density_error<1e-6 and reward_error<5e-5,(outside,density_error,reward_error)
        losses=runner.alg.update();assert all(math.isfinite(float(x)) for x in losses.values())
        assert runner.alg.optimizer.state and state_digest(runner.alg.actor.state_dict())!=initial
        # Real invalid flights must fail immediately in training AND evaluation.
        # Only motor requests are applied; no state or timer injection.
        quality_results={}
        for mode in (False, True):
            env.evaluation_mode=mode
            with torch.inference_mode():wrapped.reset()
            rejected=[]
            name='quality_eval' if mode else 'quality_train'
            with (a.output/(name+'_trace.jsonl')).open('w') as f,torch.inference_mode():
                for step in range(200):
                    sign=torch.where(torch.arange(16,device=env.device)%2==0,1.,-1.)
                    hip=sign if step%50<25 else torch.zeros_like(sign)
                    obs,_,done,info=wrapped.step(torch.stack((hip,torch.full_like(hip,-1.)),-1))
                    row=env.last_trace;f.write(json.dumps(row)+'\n')
                    for k,code in enumerate(row['failure']):
                        if code==5:
                            assert done[k] and row['flight_invalid'][k] and not row['completed'][k]
                            assert row['elapsed'][k]<15 and row['shaping_reward'][k]==0
                            assert not bool(info.get('time_outs',torch.zeros(16,device=env.device))[k])
                            rejected.append(dict(env=k,elapsed=row['elapsed'][k],count=row['front_flight_count'][k]))
                    if rejected:break
            assert rejected, 'Strict quality rejection not exercised'
            quality_results[name]=rejected
        result=dict(status='passed',engineering_only=True,fresh_random=True,initial_mean_max_forward=distance,
            initial_model_unchanged_before_ppo=True,raw_outside_limit_samples=outside,raw_logprob_max_error=density_error,
            reward_recompute_max_error=reward_error,losses={k:float(v) for k,v in losses.items()},
            distribution=distribution_metrics(runner.alg.actor.distribution),real_quality_rejections=quality_results,actor_observation_dim=26,packet_once=True,
            full_task_capability_passed=False,no_weights_or_data_transfer=True,no_scripted_data_in_ppo=True)
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');env.close()
    except BaseException:
        (a.output/'result.json').write_text(json.dumps(dict(status='failed',error=traceback.format_exc()),indent=2)+'\n');raise
    finally:app.close()


if __name__=='__main__':main()
