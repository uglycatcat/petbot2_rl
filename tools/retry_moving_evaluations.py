"""Retry only failed common-envelope evaluations, preserving prior evidence."""
import argparse, fcntl, json, pathlib, subprocess, time
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--name',required=True)
a=p.parse_args();root=a.root.resolve();out=root/'demo_project/runs'/('_'+a.name)
lock=(out/'eval.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=out/'eval_status.json';state=json.loads(status.read_text());launch=json.loads((out/'launch_status.json').read_text())
for key,entry in state.items():
    if entry['status']!='failed' or not key.endswith('_common'):continue
    name=key.rsplit('_i',1)[0];iteration=launch['iterations']-1
    target=out/('retry_'+key)
    if target.exists():raise RuntimeError(f'Inspect existing retry before continuing: {target}')
    entry['previous_attempts']=entry.get('previous_attempts',[])+[{k:v for k,v in entry.items() if k!='previous_attempts'}]
    entry.update(status='retry_running',retry_started_at=time.strftime('%FT%T%z'));status.write_text(json.dumps(state,indent=2))
    cmd=['python3',str(root/f'demo_project/runs/{name}/tools/run_guarded.py'),'--container',launch['container'],'--gpu','2','--reserve-mib','3500','--entrypoint','train_moving.py','--workdir',f'/workspace/runs/{name}','--log',str(out/('retry_'+key+'.log')),'--headless','--device','cuda:0','--num_envs','64','--evaluate','--seed','20261020','--speed-limit','0.3','--yaw-limit','1.0','--frame-min=-1.0','--frame-max','1.2','--checkpoint',f'/workspace/runs/{name}/logs/{name}/model_{iteration}.pt','--log_dir',f'/workspace/runs/_{a.name}/retry_{key}']
    code=subprocess.run(cmd).returncode;result=target/'evaluation.json'
    entry.update(status='complete' if code==0 and result.exists() else 'failed',exit_code=code,retry_finished_at=time.strftime('%FT%T%z'))
    if result.exists():entry['result']=json.loads(result.read_text())
    status.write_text(json.dumps(state,indent=2))
old=out/'eval_finished.json';backup=out/'eval_finished_before_retry.json'
if old.exists() and not backup.exists():backup.write_text(old.read_text())
old.write_text(json.dumps(dict(finished_at=time.strftime('%FT%T%z'),complete=sum(v['status']=='complete' for v in state.values()),expected=len(state),retry=True)))
