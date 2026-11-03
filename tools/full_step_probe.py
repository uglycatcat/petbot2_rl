"""Fresh full-task PPO and real single-flight-failure engineering probe.

No model or trajectory is loaded. Scripted quality probes never enter PPO.
Passing this program is not learned climbing or stopping acceptance.
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--asset-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    AppLauncher.add_app_launcher_args(parser);a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);app=AppLauncher(a).app
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from step_demo.config import PROJECT_ROOT,load_config
        from step_demo.runtime import load_training
        from step_demo.isaac_env import StepEnv
        from step_demo.ppo import StepRunner,runner_config,state_digest,distribution_metrics
        t=load_training(PROJECT_ROOT/'configs/restart19_gate.json');t['num_envs']=16
        env=StepEnv(load_config(model_dir=a.model_dir),t,a.asset_dir,a.device)
        env.trace_ids=list(range(16));wrapped=RslRlVecEnvWrapper(env)
        runner=StepRunner(wrapped,runner_config(t),{'training':t},device=a.device)
        assert not runner.alg.optimizer.state and all(x.requires_grad for x in runner.alg.actor.parameters())
        initial=state_digest(runner.alg.actor.state_dict());obs=wrapped.get_observations()
        assert obs['policy'].shape==(16,26)
        assert (env.packet[:,1]==.30).all() and (env.packet_count==1).all()
        outside=0;density_error=0.;reward_error=0.;terminals=[]
        with (a.output/'ppo_trace.jsonl').open('w') as f,torch.inference_mode():
            for step in range(t['num_steps_per_env']):
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
                    failed=row['failure'][k]>0 or (row['elapsed'][k]+1e-5>=15. and not row['completed'][k])
                    progress=row['shaping_reward'][k] if row['previous_intent'][k]<3 else 0.
                    # New landing credit is reported separately in each stage's
                    # total progress. This short probe independently checks
                    # failure frames (where all progress is zero) and no-land frames.
                    if failed:
                        expected=-15.-sum(row['restart_cost_components'][k])
                        reward_error=max(reward_error,abs(expected-row['reward'][k]))
                        assert not bool(info.get('time_outs',torch.zeros(16,device=env.device))[k])
                    elif row['previous_intent'][k]>=3:
                        expected=-sum(row['restart_cost_components'][k])+100.*row['completed'][k]
                        reward_error=max(reward_error,abs(expected-row['reward'][k]))
                terminals.extend(env.terminal_records);env.terminal_records.clear()
            runner.alg.compute_returns(obs)
        assert outside>0 and density_error<1e-6 and reward_error<3e-5
        losses=runner.alg.update();assert all(math.isfinite(float(x)) for x in losses.values())
        assert runner.alg.optimizer.state and state_digest(runner.alg.actor.state_dict())!=initial
        (a.output/'ppo_result.json').write_text(json.dumps(dict(status='passed',raw_outside_samples=outside,
            raw_density_error=density_error,reward_error=reward_error,losses={k:float(v) for k,v in losses.items()},
            distribution=distribution_metrics(runner.alg.actor.distribution),episodes=terminals),indent=2)+'\n')
        # Real motor commands exercise an impossible single-flight episode;
        # no contact, pose, phase, clearance or flight state is injected.
        found=[]
        with torch.inference_mode():wrapped.reset()
        with (a.output/'quality_trace.jsonl').open('w') as f,torch.inference_mode():
            for step in range(200):
                sign=torch.where(torch.arange(16,device=env.device)%2==0,1.,-1.)
                hip=sign if step%50<25 else torch.zeros_like(sign)
                action=torch.stack((hip,torch.full_like(hip,-1.)),-1)
                obs,reward,done,info=wrapped.step(action);row=env.last_trace;f.write(json.dumps(row)+'\n')
                for k,code in enumerate(row['failure']):
                    if code==5:
                        assert row['flight_invalid'][k] and not row['completed'][k]
                        assert abs(row['reward'][k]+15.+sum(row['restart_cost_components'][k]))<3e-5
                        assert not bool(info.get('time_outs',torch.zeros(16,device=env.device))[k])
                        found.append(dict(env=k,elapsed=row['elapsed'][k],flight_count=row['front_flight_count'][k]))
                if found:break
        assert found,'Real single-flight violation not exercised'
        result=dict(status='passed',engineering_only=True,fresh_random=True,no_pretrained_weights=True,
            no_scripted_data_in_ppo=True,raw_samples_preserved=True,raw_logprob_max_error=density_error,
            raw_outside_limit_samples=outside,reward_recompute_max_error=reward_error,
            real_quality_failure_examples=found,no_bootstrap_for_real_failure=True,
            actor_observation_dim=26,packet_once=True,full_task_capability_passed=False,
            no_weights_or_data_transfer=True)
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');env.close()
    except BaseException:
        (a.output/'result.json').write_text(json.dumps(dict(status='failed',error=traceback.format_exc()),indent=2)+'\n');raise
    finally:app.close()


if __name__=='__main__':main()
