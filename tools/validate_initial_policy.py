"""Fresh actor coverage gate: mean and stochastic actions, no learning/weights."""
import argparse
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('mean','sample'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);app=AppLauncher(a).app;env=None
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from step_demo.config import PROJECT_ROOT,load_config
        from step_demo.runtime import load_training
        from step_demo.isaac_env import StepEnv
        from step_demo.ppo import StepRunner,runner_config,state_digest,distribution_metrics
        t=load_training(PROJECT_ROOT/'configs/engineering.json');t['num_envs']=64
        env=StepEnv(load_config(model_dir=a.model_dir),t,a.asset_dir,a.device)
        env.trace_ids=list(range(64));wrapped=RslRlVecEnvWrapper(env)
        runner=StepRunner(wrapped,runner_config(t),dict(training=t),None,a.device)
        assert not runner.alg.optimizer.state
        initial=state_digest(runner.alg.actor.state_dict())
        records={};forward=[0.]*64;first_distribution=None
        obs=wrapped.get_observations()
        with torch.inference_mode(),(a.output/'trace.jsonl').open('x') as file:
            for k in range(env.max_episode_length+1):
                action=runner.alg.actor(obs,stochastic_output=a.mode=='sample')
                if a.mode=='sample' and first_distribution is None:
                    first_distribution=distribution_metrics(runner.alg.actor.distribution)
                obs,_,dones,_=wrapped.step(action);runner.alg.actor.reset(dones)
                row=env.last_trace;file.write(json.dumps(row)+'\n')
                for i in range(64):
                    if i not in records and row['task']['failure'][i]==0:
                        forward[i]=max(forward[i],row['com'][i][0])
                for r in env.terminal_records: records.setdefault(r['env'],r)
                env.terminal_records.clear()
                if len(records)==64:break
        assert len(records)==64
        assert state_digest(runner.alg.actor.state_dict())==initial
        reached=sum(x>=.10 for x in forward)
        data=dict(status='passed' if reached>=58 else 'failed',mode=a.mode,
            initialization='random',optimizer_states=0,weights_loaded=False,used_for_training=False,
            actor_sha256=initial,initial_distribution=first_distribution,
            reached_10cm_before_failure=reached,required=58,episodes=records,max_forward=forward)
        (a.output/'result.json').write_text(json.dumps(data,indent=2)+'\n')
        if reached<58:raise RuntimeError(f'Initial policy coverage {reached}/64: diagnose before short training')
    except BaseException:
        (a.output/'error.txt').write_text(traceback.format_exc())
        if not (a.output/'result.json').exists():
            (a.output/'result.json').write_text(json.dumps(dict(status='failed',error=traceback.format_exc()),indent=2)+'\n')
        raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
