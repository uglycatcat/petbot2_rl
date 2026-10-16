"""Launch this authorized batch once, only after a recorded gate. No automatic retries."""
import argparse,fcntl,json,os,pathlib,shutil,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
root=a.root.resolve();src=root/'demo_project/upright_rl_demo';runs=root/'demo_project/runs';out=runs/'_upright_v2';out.mkdir(exist_ok=True)
lock=(out/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=out/'launch_status.json'
if status.exists():raise RuntimeError('Already launched or attempted. Inspect existing jobs; no automatic retries')
gate=json.loads((out/'gate.json').read_text());assert gate['passed']
def git(*args):return subprocess.check_output(['git','-C',str(src),*args],text=True).strip()
assert not git('status','--porcelain'), 'Source dirty'
commit=git('rev-parse','HEAD');assert not git('diff',gate['tested_commit'],commit,'--','upright','train.py'), 'Task differs from tested code'
state={'commit':commit,'gate':gate,'started_at':time.strftime('%FT%T%z'),'train':{}}
def save():status.write_text(json.dumps(state,indent=2))
# Check both devices before creating either job.
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
mem={int(i):(int(t),int(u)) for i,t,u in (line.split(',') for line in rows.splitlines())}
for gpu,reserve in [(1,4500),(0,4000)]:
 total,used=mem[gpu];assert used+reserve<total*.78, (gpu,used,reserve)
for seed in [42,43]:
 name=f'upright_v2_s{seed}';dst=runs/name
 if dst.exists():raise RuntimeError(f'Existing run {name}; inspect before any restart')
 subprocess.run(['git','clone','--no-local',str(src),str(dst)],env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'},check=True)
 subprocess.run(['git','-C',str(dst),'checkout','--detach',commit],check=True)
 shutil.copytree(src/'artifacts/petbot_asset',dst/'artifacts/petbot_asset')
save()
for seed,gpu,reserve in [(42,1,4500),(43,0,4000)]:
 name=f'upright_v2_s{seed}';dst=runs/name;log=out/f'{name}.log'
 command=['python3',str(dst/'tools/run_guarded.py'),'--gpu',str(gpu),'--reserve-mib',str(reserve),
  '--workdir',f'/workspace/runs/{name}','--log',str(log),'--headless','--device','cuda:0','--num_envs','2048',
  '--iterations','3000','--seed',str(seed),'--log_dir',f'/workspace/runs/{name}/logs/{name}']
 with (out/f'{name}_guard.log').open('w') as f:proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 state['train'][name]={'pid':proc.pid,'gpu':gpu,'status':'initializing','command':command};save()
 deadline=time.monotonic()+300
 while time.monotonic()<deadline:
  if proc.poll() is not None:raise RuntimeError(f'{name} exited {proc.returncode}; inspect guard/logs')
  if log.exists() and 'Learning iteration' in log.read_text():break
  time.sleep(5)
 else:raise RuntimeError(f'{name}: no PPO by startup deadline; inspect live process, do not repeat launch')
 state['train'][name]['status']='ppo_started';save()
with (out/'eval_queue.log').open('w') as f:
 proc=subprocess.Popen(['python3',str(runs/'upright_v2_s42/tools/evaluate_v2.py'),'--root',str(root)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
state['eval_pid']=proc.pid;state['all_ppo_at']=time.strftime('%FT%T%z');save()
print(json.dumps(state,indent=2))
