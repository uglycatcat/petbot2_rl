"""Finite action replay and rear-transition intervention, never training data."""
import argparse
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    rows=json.loads(a.trajectory.read_text());a.output.mkdir(parents=True,exist_ok=False)
    cases=[dict(kind='original')]+[dict(kind='tail',hip=h,wheel=w,rate=r)
        for h in (-.8,-.5,0.,.5) for w in (4.,8.,12.) for r in (3.,20.)]
    app=AppLauncher(a).app
    try:
        import torch
        from step_demo.config import load_config,PROJECT_ROOT
        from step_demo.isaac_env import StepEnv,tensor
        training=json.loads((PROJECT_ROOT/'configs/round03_seed71.json').read_text())
        training['num_envs']=len(cases)
        env=StepEnv(load_config(model_dir=a.model_dir),training,a.asset_dir,a.device)
        env.trace_ids=list(range(len(cases)));env.reset()
        results=[dict(case=c,prefix_max_com_error=0.,landed_before_intervention=False,terminal=None,
                      ever_landed=False,ever_all_top=False) for c in cases]
        requested=[0.]*len(cases)
        original=env._capture
        with (a.output/'substeps.jsonl').open('x') as sub, (a.output/'trace.jsonl').open('x') as trace:
            def capture(self,counter):
                old=self._capture_counter;original(counter)
                if self._capture_counter==old:return
                forces={name:tensor(sensor.data.net_forces_w)[:,0] for name,sensor in self.contacts}
                net=torch.stack([forces[name] for name in self.wheel_names],dim=1)
                sub.write(json.dumps(dict(counter=counter,episode=self.episode_serial.tolist(),
                    failure=self.failure.tolist(),rear_air_s=self.rear_air_s.tolist(),
                    net_force=net.tolist(),platform_normal=self.platform_normal.tolist(),
                    wheel_z=self.minimum[:,:,2].tolist(),wheel_min_x=(self.minimum[:,:,0]-self.edge_x[:,None]).tolist(),
                    hip_target=self.hip_target.tolist(),wheel_target=self.wheel_target.tolist(),
                    hip_actual=tensor(self.robot.data.joint_pos)[:,self.hips].tolist(),
                    wheel_actual=tensor(self.robot.data.joint_vel)[:,self.wheels].tolist()))+'\n')
            env._capture=types.MethodType(capture,env)
            with torch.inference_mode():
                for step in range(250):
                    values=[];t=step*env.step_dt
                    for i,c in enumerate(cases):
                        if step<49 or c['kind']=='original' and step<len(rows):
                            value=rows[step]['action'];requested[i]=rows[step]['hip_target']
                        elif c['kind']=='original':value=[0.,-1.]
                        else:
                            target=c['hip'] if t<2.5 else 0.
                            requested[i]+=max(-c['rate']*env.step_dt,min(c['rate']*env.step_dt,target-requested[i]))
                            value=[requested[i]/1.2 if requested[i]>=0 else requested[i],
                                   (c['wheel'] if t<3.7 else 0.)/12.5-1.]
                        values.append(value)
                    obs,reward,_,_,_=env.step(torch.tensor(values,device=env.device))
                    assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                    d=env.last_trace;trace.write(json.dumps(d)+'\n')
                    for i,res in enumerate(results):
                        if d['episode'][i]!=1 or res['terminal'] is not None:continue
                        if step<49:
                            res['prefix_max_com_error']=max(res['prefix_max_com_error'],
                                max(abs(x-y) for x,y in zip(d['com'][i],rows[step]['com'])))
                            res['landed_before_intervention']|=all(d['front_landed'][i]) and not d['failure'][i]
                        res['ever_landed']|=all(d['front_landed'][i]);res['ever_all_top']|=all(d['supported_top'][i])
                        if d['failure'][i] or d['completed'][i]:
                            res['terminal']=dict(time=d['elapsed'][i],failure=d['failure'][i],
                                completed=d['completed'][i],phase=d['phase'][i],com=d['com'][i])
        (a.output/'result.json').write_text(json.dumps(dict(status='passed',kind='finite_diagnostic_not_training_data',
            source_sha256=env.asset['source_sha256'],cases=results),indent=2)+'\n')
        env.close();print(json.dumps(results))
    except BaseException:
        import traceback
        (a.output/'failure.txt').write_text(traceback.format_exc());raise
    finally:app.close()


if __name__=='__main__':main()
