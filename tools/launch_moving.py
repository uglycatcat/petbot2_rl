"""One-shot fixed-version moving-task batch. Existing state requires manual recovery."""
import argparse,fcntl,json,os,pathlib,re,shutil,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--name',required=True);p.add_argument('--iterations',type=int,default=1500);a=p.parse_args()
assert re.fullmatch('[a-z0-9_]+',a.name) and a.iterations>=751
root=a.root.resolve();source=root/'demo_project/upright_rl_demo';runs=root/'demo_project/runs';out=runs/('_'+a.name);out.mkdir(exist_ok=True)
lock=(out/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
path=out/'launch_status.json'
if path.exists():raise RuntimeError('Existing batch: inspect logs/PIDs/checkpoints; do not restart blindly')
gate=json.loads((out/'gate.json').read_text());assert gate['engineering_passed']
def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
assert not git('status','--porcelain');commit=git('rev-parse','HEAD')
assert not git('diff',gate['tested_commit'],commit,'--','upright','train_moving.py')
container='petbot-upright-night';assert subprocess.check_output(['docker','inspect',container,'--format','{{.State.Running}}'],text=True).strip()=='true'
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
for line in rows.splitlines():
 gpu,total,used=map(int,line.split(','));assert used+4000<total*.78,(gpu,used,total)
state=dict(name=a.name,commit=commit,container=container,thread=os.environ.get('CODEX_THREAD_ID'),started_at=time.strftime('%FT%T%z'),iterations=a.iterations,checkpoints=[250,750,a.iterations-1],gate=gate,train={})
def save():path.write_text(json.dumps(state,indent=2))
for label,seed,gpu,speed,yaw,lo,hi in [('b',42,2,.3,1.,-1.,1.2),('b',43,3,.3,1.,-1.,1.2),('a',42,0,.1,.3,-.3,.3),('a',43,1,.1,.3,-.3,.3)]:
 name=f'{a.name}_{label}{seed}';dst=runs/name
 if dst.exists():raise RuntimeError(f'Existing run: {name}')
 subprocess.run(['git','clone','--no-local',str(source),str(dst)],env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'},check=True)
 subprocess.run(['git','-C',str(dst),'checkout','--detach',commit],check=True)
 shutil.copytree(source/'artifacts/petbot_asset',dst/'artifacts/petbot_asset')
 envelope=['--speed-limit',str(speed),'--yaw-limit',str(yaw),'--frame-min',str(lo),'--frame-max',str(hi)]
 command=['python3',str(dst/'tools/run_guarded.py'),'--container',container,'--gpu',str(gpu),'--reserve-mib','4000','--entrypoint','train_moving.py','--workdir',f'/workspace/runs/{name}','--log',str(out/(name+'.log')),'--headless','--device','cuda:0','--num_envs','2048','--iterations',str(a.iterations),'--seed',str(seed),*envelope,'--log_dir',f'/workspace/runs/{name}/logs/{name}']
 with (out/(name+'_guard.log')).open('w') as f:proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 state['train'][name]=dict(pid=proc.pid,gpu=gpu,envelope=envelope,status='initializing',command=command);save()
 deadline=time.monotonic()+600;log=out/(name+'.log')
 while time.monotonic()<deadline:
  if proc.poll() is not None:raise RuntimeError(f'{name} exited: {proc.returncode}')
  if log.exists() and 'Learning iteration' in log.read_text(errors='replace'):break
  time.sleep(5)
 else:raise RuntimeError('PPO startup timeout: inspect existing process')
 state['train'][name]['status']='ppo_started';save()
with (out/'eval_queue.log').open('w') as f:proc=subprocess.Popen(['python3',str(dst/'tools/evaluate_moving.py'),'--root',str(root),'--name',a.name],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
state.update(eval_pid=proc.pid,all_ppo_at=time.strftime('%FT%T%z'));save()
