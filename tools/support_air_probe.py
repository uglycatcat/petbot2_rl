"""Finite PhysX airborne counterexample; not a training reset or PPO data."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);app=AppLauncher(a).app
    try:
        import torch
        from step_demo.config import load_config,PROJECT_ROOT
        from step_demo.isaac_env import StepEnv,tensor
        t=json.loads((PROJECT_ROOT/'configs/round09_seed70.json').read_text());t['num_envs']=8
        env=StepEnv(load_config(model_dir=a.model_dir),t,a.asset_dir,a.device)
        env.trace_ids=list(range(8));env.reset()
        pose=tensor(env.robot.data.default_root_pose).clone();pose[:,:3]+=env.scene.env_origins
        pose[4:,2]+=.08  # Four true-flight cases, four unchanged grounded controls.
        env.robot.write_root_pose_to_sim_index(root_pose=pose)
        failures={}
        with (a.output/'trace.jsonl').open('x') as f:
            for step in range(10):
                env.step(torch.tensor([[0.,-1.]]*8,device=env.device))
                d=env.last_trace;f.write(json.dumps(d)+'\n')
                for i in range(8):
                    if d['episode'][i]==1 and d['failure'][i]:
                        failures.setdefault(i,dict(failure=d['failure'][i],elapsed=d['elapsed'][i],
                            clearance=d['rear_clearance'][i],air_s=d['rear_air_s'][i]))
                assert not any(i<4 for i in failures), 'Grounded control falsely terminated'
                if len(failures)==4:break
        assert set(failures)==set(range(4,8)) and all(r['failure']==2 for r in failures.values())
        assert all(min(r['clearance'])>.004 and max(r['air_s'])>.02 for r in failures.values())
        (a.output/'result.json').write_text(json.dumps(dict(status='passed',
            kind='isolated_physical_counterexample_not_training',ground_controls=4,flight_cases=failures),indent=2)+'\n')
        env.close()
    finally:app.close()


if __name__=='__main__':main()
