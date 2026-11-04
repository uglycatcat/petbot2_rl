"""Finite motor-only reachability study, never a source of PPO training data.

All cases start at the original default scene. Only two symmetric motor
commands change. No pose, contact, timer, terrain, or stage state is injected.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
        from step_demo.config import PROJECT_ROOT,load_config
        from step_demo.runtime import load_training
        from step_demo.isaac_env import StepEnv
        cases=[dict(wheel=w,delay=d,hip=h,duration=t) for w,d,h,t in
               itertools.product((8.,12.,16.,20.),(.2,.3,.4,.5),(-.4,-.7,-1.),(.08,.16,.30))]
        t=load_training(PROJECT_ROOT/'configs/restart21_gate.json');t.update(num_envs=len(cases),_evaluation=True)
        env=StepEnv(load_config(model_dir=a.model_dir),t,a.asset_dir,a.device)
        env.trace_ids=list(range(len(cases)));env.reset()
        stats=[dict(case=c,terminal=None,cleared=False,landed=False,four_top=False,
                    max_pitch=0.,max_pitch_rate=0.,max_forward=0.,max_front_z=0.) for c in cases]
        with (a.output/'trace.jsonl').open('w') as f,torch.inference_mode():
            for k in range(750):
                elapsed=k*env.step_dt
                values=[]
                for c in cases:
                    hip=c['hip'] if c['delay']<=elapsed<c['delay']+c['duration'] else 0.
                    values.append([hip,c['wheel']/12.5-1.])
                obs,reward,done,timeout,_=env.step(torch.tensor(values,device=env.device))
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                row=env.last_trace;f.write(json.dumps(row)+'\n')
                for i,s in enumerate(stats):
                    if s['terminal'] is not None:continue
                    assert row['episode'][i]==1
                    s['cleared'] |= all(row['front_cleared'][i]);s['landed'] |= all(row['front_landed'][i])
                    s['four_top'] |= all(row['supported_top'][i])
                    s['max_pitch']=max(s['max_pitch'],abs(row['world_pitch'][i]))
                    s['max_pitch_rate']=max(s['max_pitch_rate'],abs(row['world_pitch_rate'][i]))
                    s['max_forward']=max(s['max_forward'],row['com'][i][0])
                    s['max_front_z']=max(s['max_front_z'],min(x[2] for x in row['wheel_min'][i][:2]))
                    if done[i] or timeout[i]:
                        s['terminal']=dict(failure=row['failure'][i],t=row['elapsed'][i],
                            flight=row['front_flight_count'][i],invalid=row['flight_invalid'][i],
                            success=row['completed'][i],forward=row['com'][i][0])
                if all(x['terminal'] is not None for x in stats):break
        assert all(x['terminal'] is not None for x in stats)
        result=dict(status='passed',engineering_only=True,no_weights_loaded=True,no_data_for_ppo=True,
                    cases=stats,cleared=sum(x['cleared'] for x in stats),landed=sum(x['landed'] for x in stats),
                    four_top=sum(x['four_top'] for x in stats),source_sha256=env.asset['source_sha256'])
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');env.close()
    except BaseException:
        (a.output/'result.json').write_text(json.dumps(dict(status='failed',error=traceback.format_exc()),indent=2)+'\n');raise
    finally:app.close()


if __name__=='__main__':main()
