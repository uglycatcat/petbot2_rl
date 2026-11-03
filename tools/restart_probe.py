"""Finite container physics gate for auxiliary initial states and real hold.

Scripted actions here are engineering probes only, never training data.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    app=AppLauncher(a).app
    try:
        import torch
        from step_demo.config import load_config,PROJECT_ROOT
        from step_demo.runtime import load_training
        from step_demo.isaac_env import StepEnv,tensor
        t=load_training(PROJECT_ROOT/'configs/restart15_gate.json')
        t.update(num_envs=16,hold_after_success_s=10.)
        env=StepEnv(load_config(model_dir=a.model_dir),t,a.asset_dir,a.device)
        env.trace_ids=list(range(16));obs,_=env.reset()
        assert obs['policy'].shape==(16,26)
        assert torch.all(env.estimator.velocity==0), 'Estimator received physical initial speed'
        assert (env.packet[:,1]==-.45).all()
        assert not env.front_lifted.any() and not env.front_cleared.any() and not env.front_landed.any()
        first=None;initial_pose=tensor(env.robot.data.root_pos_w).clone()
        with (a.output/'trace.jsonl').open('w') as file:
            for step in range(200):
                speed=(torch.arange(16,device=env.device)%4).float()*.2
                wheel=(speed/.03)*(max(0.,1.-step*.02/.5))
                actions=torch.stack((torch.zeros_like(wheel),wheel/12.5-1.),-1)
                obs,reward,done,truncated,_=env.step(actions)
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                if step==0:first=env.last_trace
                file.write(json.dumps(env.last_trace)+'\n')
        # Static group proves ordinary top support and continued physics hold.
        static=torch.arange(0,16,4,device=env.device)
        assert (env.episode_serial[static]==1).all()
        assert env.supported_top[static].all()
        assert (env.auxiliary_hold[static]>2.5).all()
        assert (env.flight.count[static]==0).all()
        assert env.captured_substeps==2000
        assert not env.front_landed.any(), 'Auxiliary task must not fabricate climb history'
        result=dict(status='passed',probe_only=True,no_training_data=True,
            actor_dim=26,initial_physical_speed=[x[0] for x in first['velocity']],
            initial_sensor_estimated_speed=[x[0] for x in first['estimated_velocity']],
            final_hold=env.auxiliary_hold.tolist(),totals=env.totals,
            captured_substeps=env.captured_substeps,static_group_preserved=True,
            moving_group_outcomes_require_behavior_review=True)
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        env.close()
    finally:
        app.close()


if __name__=='__main__':main()
