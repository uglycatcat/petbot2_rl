"""One-shot four-GPU round deployment after explicit engineering gate."""
import argparse,datetime,fcntl,json,os,pathlib,shutil,subprocess,time
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--round',choices=('v3','v4','v5','v6'),required=True);p.add_argument('--iterations',type=int,default=1500);a=p.parse_args()
root=a.root.resolve();src=root/'demo_project/upright_rl_demo';runs=root/'demo_project/runs';out=runs/('_upright_'+a.round);out.mkdir(exist_ok=True)
lock=(out/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);status=out/'launch_status.json'
if status.exists():raise RuntimeError('Existing round: inspect jobs before any repeat')
start_record=json.loads((out/'round_start.json').read_text())
cutoff=datetime.datetime(2026,10,17,8,tzinfo=ZoneInfo('Asia/Shanghai'))
assert datetime.datetime.fromisoformat(start_record['started_at'])<cutoff, 'No new round after final 08:00 trigger'
gate=json.loads((out/'gate.json').read_text());assert gate['engineering_passed']
def git(*args):return subprocess.check_output(['git','-C',str(src),*args],text=True).strip()
assert not git('status','--porcelain');commit=git('rev-parse','HEAD')
assert not git('diff',gate['tested_commit'],commit,'--','upright','train.py')
container='petbot-upright-night'
assert subprocess.check_output(['docker','inspect',container,'--format','{{.State.Running}}'],text=True).strip()=='true'
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
mem={int(i):(int(t),int(u)) for i,t,u in (line.split(',') for line in rows.splitlines())}
for gpu in range(4):assert mem[gpu][1]+4000<mem[gpu][0]*.78,(gpu,mem[gpu])
state={'thread':os.environ.get('CODEX_THREAD_ID'),'round':a.round,'commit':commit,'container':container,'started_at':time.strftime('%FT%T%z'),'iterations':a.iterations,
 'checkpoints':sorted(set([250,750,a.iterations-1])),'gate':gate,'train':{}}
def save():status.write_text(json.dumps(state,indent=2))
# Long control-step variants launch first, then GPU0/1 variants.
for label,seed,gpu,decimation in [('b',42,2,30),('b',43,3,30),('a',42,0,10),('a',43,1,10)]:
 name=f'upright_{a.round}_{label}{seed}';dst=runs/name
 if dst.exists():raise RuntimeError(f'Existing run {name}: inspect, do not overwrite')
 subprocess.run(['git','clone','--no-local',str(src),str(dst)],env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'},check=True)
 subprocess.run(['git','-C',str(dst),'checkout','--detach',commit],check=True)
 shutil.copytree(src/'artifacts/petbot_asset',dst/'artifacts/petbot_asset')
 log=out/f'{name}.log'
 command=['python3',str(dst/'tools/run_guarded.py'),'--container',container,'--gpu',str(gpu),'--reserve-mib','4000','--workdir',f'/workspace/runs/{name}',
  '--log',str(log),'--headless','--device','cuda:0','--num_envs','2048','--iterations',str(a.iterations),'--seed',str(seed),'--control-decimation',str(decimation),'--log_dir',f'/workspace/runs/{name}/logs/{name}']
 with (out/f'{name}_guard.log').open('w') as f:proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 state['train'][name]={'pid':proc.pid,'gpu':gpu,'decimation':decimation,'status':'initializing','command':command};save()
 deadline=time.monotonic()+600
 while time.monotonic()<deadline:
  if proc.poll() is not None:raise RuntimeError(f'{name} exit {proc.returncode}; inspect logs')
  if log.exists() and 'Learning iteration' in log.read_text():break
  time.sleep(5)
 else:raise RuntimeError('No PPO before deadline: inspect live job; no automatic restart')
 state['train'][name]['status']='ppo_started';save()
with (out/'eval_queue.log').open('w') as f:
 proc=subprocess.Popen(['python3',str(dst/'tools/evaluate_round.py'),'--root',str(root),'--round',a.round],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
state['eval_pid']=proc.pid;state['all_ppo_at']=time.strftime('%FT%T%z');save()
