"""Finite PhysX action-authority comparison; never a demonstration/training source."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def profiles():
    cases=[]
    for hip in (-1., -.5, .5, 1.):
        for wheel in (0., 8., 16., 25.):
            cases.append(dict(kind='direct',preload=0.,hip=hip,wheel=wheel))
    for preload,hip in ((-1.,1.),(1.,-1.),(-.5,1.),(.5,-1.)):
        for wheel in (0.,8.,16.,25.):
            cases.append(dict(kind='switch',preload=preload,hip=hip,wheel=wheel))
    for delay in (.15,.25,.35,.45):
        for wheel in (8.,16.,25.):
            cases.append(dict(kind='delayed',preload=0.,hip=-1.,wheel=wheel,delay=delay))
    return cases


def edge_profiles():
    # Finite open-loop reachability checks, never used to initialize or train PPO.
    return [dict(kind='pulse', preload=preload, hip=-1., wheel=wheel,
                 delay=distance/(.03*wheel)+.05, duration=duration)
            for preload in (0., .5)
            for wheel in (8., 12., 16., 20.)
            for distance in (.07, .10, .13, .16)
            for duration in (.08, .16, .30)]


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--authority',choices=('baseline','hip_fast','wheel_fast','fast'),required=True)
    p.add_argument('--hip-rate',type=float,help='Finite software target slew comparison; physical limits unchanged')
    p.add_argument('--profile-set',choices=('lift','edge'),default='lift')
    AppLauncher.add_app_launcher_args(p);args=p.parse_args()
    if args.hip_rate is not None and not 0<args.hip_rate<=100.:p.error('hip-rate must be in (0,100]')
    args.output.mkdir(parents=True,exist_ok=False)
    app=AppLauncher(args).app
    try:
        import torch
        from step_demo.config import PROJECT_ROOT,load_config
        from step_demo.isaac_env import StepEnv
        cases=edge_profiles() if args.profile_set=='edge' else profiles()
        training=json.loads((PROJECT_ROOT/'configs/train_v1.json').read_text())
        training['num_envs']=len(cases)
        if args.authority in ('fast','hip_fast'):training['hip_target_rate_rad_s']=100.
        if args.authority in ('fast','wheel_fast'):training['wheel_target_accel_rad_s2']=1000.
        if args.hip_rate is not None:training['hip_target_rate_rad_s']=args.hip_rate
        env=StepEnv(load_config(model_dir=args.model_dir),training,args.asset_dir,args.device)
        env.trace_ids=list(range(len(cases)));env.reset()
        result=[dict(case=i,profile=c,max_both_front_min_z=-1.,lifted=False,cleared=False,landed=False,
                     peak_actual_wheel=0.,terminal=None) for i,c in enumerate(cases)]
        with (args.output/'trace.jsonl').open('x') as file,torch.inference_mode():
            for step in range(250 if args.profile_set=='edge' else 150):
                t=step*env.step_dt;values=[]
                for c in cases:
                    if t<.10: hip,wheel=0.,0.
                    elif c['kind']=='pulse':
                        hip=(c['preload'] if t<.1+c['delay'] else
                             c['hip'] if t<.1+c['delay']+c['duration'] else 0.)
                        wheel=c['wheel']
                    elif c['kind']=='switch' and t<.45:hip,wheel=c['preload'],0.
                    elif c['kind']=='delayed' and t<.1+c['delay']:hip,wheel=0.,c['wheel']
                    else:hip,wheel=c['hip'],c['wheel']
                    values.append([hip,wheel/12.5-1.])
                obs,reward,terminated,truncated,_=env.step(torch.tensor(values,device=env.device))
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                row=env.last_trace;file.write(json.dumps(dict(step=step,**row))+'\n')
                for i,r in enumerate(result):
                    if row['episode'][i]!=1 or r['terminal'] is not None:continue
                    r['max_both_front_min_z']=max(r['max_both_front_min_z'],min(w[2] for w in row['wheel_min'][i][:2]))
                    for field in ('lifted','cleared','landed'):r[field]|=all(row['front_'+field][i])
                    r['peak_actual_wheel']=max(r['peak_actual_wheel'],*(abs(row['joint_velocity'][i][j]) for j in env.wheels))
                    if row['failure'][i] or row['completed'][i]:
                        r['terminal']=dict(time=row['elapsed'][i],failure=row['failure'][i],completed=row['completed'][i],com=row['com'][i])
        summary=dict(status='passed',kind='physical_authority_diagnostic_not_training_data',authority=args.authority,profile_set=args.profile_set,
            training=training,cases=result,lifted=sum(x['lifted'] for x in result),cleared=sum(x['cleared'] for x in result),
            landed=sum(x['landed'] for x in result),max_front_height=max(x['max_both_front_min_z'] for x in result),
            source_sha256=env.asset['source_sha256'],captured_substeps=env.captured_substeps)
        (args.output/'result.json').write_text(json.dumps(summary,indent=2)+'\n');env.close()
        print(json.dumps({k:v for k,v in summary.items() if k not in ('cases','training')},ensure_ascii=False))
    except BaseException:
        import traceback
        (args.output/'failure.txt').write_text(traceback.format_exc());raise
    finally:app.close()


if __name__=='__main__':main()
