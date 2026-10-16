"""Single, locked checkpoint evaluation queue. Never launches or resumes training."""
import argparse,fcntl,json,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
root=a.root.resolve();out=root/'demo_project/runs/_upright_v2';out.mkdir(exist_ok=True)
lock=(out/'eval.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=out/'eval_status.json'
if status.exists():raise RuntimeError('Existing queue status; inspect instead of repeating jobs')
state={}
for _ in range(48*360):
 remaining=False
 for iteration in (250,750,1500,2999):
  for seed in (42,43):
   run=f'upright_v2_s{seed}';key=f'{run}_i{iteration}'
   if state.get(key,{}).get('status') in ('complete','failed'):continue
   remaining=True;checkpoint=root/f'demo_project/runs/{run}/logs/{run}/model_{iteration}.pt'
   if not checkpoint.exists() or time.time()-checkpoint.stat().st_mtime<20:continue
   total,used=map(int,subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True).split(','))
   if used+3500>total*.78:continue
   state[key]={'status':'running'};status.write_text(json.dumps(state,indent=2))
   command=['python3',str(pathlib.Path(__file__).with_name('run_guarded.py')),'--gpu','1','--reserve-mib','3500',
     '--workdir',f'/workspace/runs/{run}','--log',str(out/f'eval_{key}.log'),'--headless','--device','cuda:0',
     '--num_envs','64','--evaluate','--seed','20261016','--checkpoint',f'/workspace/runs/{run}/logs/{run}/model_{iteration}.pt',
     '--log_dir',f'/workspace/runs/_upright_v2/eval_{key}']
   rc=subprocess.run(command).returncode;result=out/f'eval_{key}/evaluation.json'
   state[key].update(status='complete' if rc==0 and result.exists() else 'failed',exit_code=rc)
   if result.exists():state[key]['result']=json.loads(result.read_text())
   status.write_text(json.dumps(state,indent=2))
 if not remaining:break
 time.sleep(10)
