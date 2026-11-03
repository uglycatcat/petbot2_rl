"""Real-physics check of fresh STOP initialization and PPO raw-action storage.

The untrained actor and one-update probe are diagnostic only; neither weights
nor trajectories are exported to the subsequent independent training runs.
"""
import argparse
import json
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
        t=load_training(PROJECT_ROOT/'configs/restart18_gate.json');t['num_envs']=16
        actual=dict(t,_evaluation=True,hold_after_success_s=2.)
        env=StepEnv(load_config(model_dir=a.model_dir),actual,a.asset_dir,a.device);env.trace_ids=list(range(16))
        wrapped=RslRlVecEnvWrapper(env);runner=StepRunner(wrapped,runner_config(t),{'training':t},device=a.device)
        assert not runner.alg.optimizer.state and all(p.requires_grad for p in runner.alg.actor.parameters())
        initial_sha=state_digest(runner.alg.actor.state_dict());policy=runner.get_inference_policy(a.device)
        obs=wrapped.get_observations();records={}
        with (a.output/'initial_policy_trace.jsonl').open('w') as file,torch.inference_mode():
            for step in range(env.max_episode_length+1):
                raw=policy(obs);obs,_,_,_=wrapped.step(raw);file.write(json.dumps(env.last_trace)+'\n')
                for record in env.terminal_records:records.setdefault(record['env'],record)
                env.terminal_records.clear()
                if len(records)==16:break
        assert len(records)==16
        assert state_digest(runner.alg.actor.state_dict())==initial_sha
        assert not runner.alg.optimizer.state
        (a.output/'initial_policy_result.json').write_text(json.dumps(dict(
            episodes=list(records.values()),successes=sum(r['success'] for r in records.values()),
            initial_actor_sha256=initial_sha,optimizer_states=0),indent=2)+'\n')
        # Actual native PPO transition/storage, with unchanged source-limit
        # clipping only at the environment. Restore no model or rollout.
        env.evaluation_mode=False;env.extra_hold_s=0.
        # Estimator tensors were produced during inference above. Reset in
        # the same mode, as native rollout collection does for auto-resets.
        with torch.inference_mode():
            wrapped.reset();obs=wrapped.get_observations()
        runner.alg.actor.train();runner.alg.critic.train();outside=0;density_error=0.
        with (a.output/'ppo_trace.jsonl').open('w') as file,torch.inference_mode():
            for step in range(t['num_steps_per_env']):
                raw=runner.alg.act(obs);before=raw.clone();logprob=runner.alg.actor.get_output_log_prob(raw).clone()
                obs,reward,done,info=wrapped.step(raw);file.write(json.dumps(env.last_trace)+'\n')
                assert torch.equal(raw,before)
                torch.testing.assert_close(torch.tensor(env.last_trace['action'],device=env.device),before,rtol=0,atol=0)
                torch.testing.assert_close(torch.tensor(env.last_trace['requested_action'],device=env.device),before.clamp(-1,1),rtol=0,atol=0)
                outside+=int((before.abs()>1).sum())
                runner.alg.process_env_step(obs,reward,done,info)
                torch.testing.assert_close(runner.alg.storage.actions[step],before,rtol=0,atol=0)
                error=(runner.alg.storage.actions_log_prob[step].squeeze(-1)-logprob).abs().max().item()
                density_error=max(density_error,error);assert error<1e-6
                assert torch.isfinite(reward).all()
            runner.alg.compute_returns(obs)
        assert outside>0
        losses=runner.alg.update();assert all(__import__('math').isfinite(float(v)) for v in losses.values())
        assert runner.alg.optimizer.state and state_digest(runner.alg.actor.state_dict())!=initial_sha
        result=dict(status='passed',engineering_only=True,initialization='fresh_random_network_with_default_target_bias',
            initial_policy_episodes=list(records.values()),initial_policy_successes=sum(r['success'] for r in records.values()),
            initial_actor_sha256=initial_sha,no_optimizer_before_probe=True,no_historical_weights=True,
            raw_samples_preserved_in_ppo=True,raw_logprob_max_error=density_error,raw_outside_limit_samples=outside,
            clipping_is_existing_actuator_limit=True,one_physical_rollout_updates=1,finite_losses={k:float(v) for k,v in losses.items()},
            distribution=distribution_metrics(runner.alg.actor.distribution),captured_substeps=env.captured_substeps,
            can_transfer_weights_or_data=False)
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');env.close()
    except BaseException:
        (a.output/'result.json').write_text(json.dumps(dict(status='failed',error=traceback.format_exc()),indent=2)+'\n');raise
    finally:app.close()


if __name__=='__main__':main()
