"""Reconcile actual exits, resume fixed checkpoints, and finish one existing batch."""
import argparse,fcntl,json,pathlib,subprocess,time,shutil,os
p=argparse.ArgumentParser();p.add_argument('plan',type=pathlib.Path);p.add_argument('--reserve-mib',type=int,default=8192);p.add_argument('--exclude-gpus',default='');a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[3];plan=json.loads(a.plan.read_text());out=root/'demo_project/runs'/('_'+plan['name']);path=out/'launch_status.json'
lock=(root/'demo_project/runs/upright_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=json.loads(path.read_text());assert state['commit']==plan['commit'];assert not state.get('eval_pid')
excluded={int(x) for x in a.exclude_gpus.split(',') if x};state['excluded_gpus']=sorted(excluded)
archive=out/('resume_archive_'+time.strftime('%H%M%S'));archive.mkdir();state['recovery_archive']=archive.name

def save():
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(path)
def live(pid):
 f=pathlib.Path('/proc')/str(pid)/'cmdline';return f.exists() and b'run_guarded.py' in f.read_bytes()
def scan(permit_existing_failure=False):
 for name,j in state['train'].items():
  if j['status'] in ('pending','resume_pending','complete'):continue
  end=out/(name+'.exit.json')
  if end.exists():
   result=json.loads(end.read_text())
   if result['exit_code']==0 and (root/f'demo_project/runs/{name}/logs/{name}/model_599.pt').exists():j['status']='complete'
   elif permit_existing_failure:j['status']='resume_pending'
   else:
    j['status']='interrupted_needs_resume';save();raise RuntimeError('New interruption: '+name+'; stop allocations and inspect')
  elif not live(j.get('pid',0)):
   save();raise RuntimeError('Guard missing without exit: '+name)
 save()
for v in plan['variants']:
 name=v['run']
 if name not in state['train'] or not state['train'][name].get('pid'):
  state['train'][name]=dict(pid=0,gpu=v['gpu'],status='pending',num_envs=v['envs'],envelope=plan['envelope']+['--wheel-slew',str(v['wheel_slew']),'--frame-slew',str(v['frame_slew'])])
scan(permit_existing_failure=True)
# Freeze all prior recovery evidence before any checkpoint can be overwritten.
for name,j in state['train'].items():
 if j['status']!='resume_pending':continue
 logdir=root/f'demo_project/runs/{name}/logs/{name}';checkpoints=sorted(logdir.glob('model_*.pt'),key=lambda f:int(f.stem.split('_')[-1]))
 if not checkpoints:raise RuntimeError('Interrupted training without checkpoint: '+name)
 dest=archive/name;dest.mkdir();shutil.copy2(logdir/'config.json',dest/'config.json')
 for f in checkpoints:shutil.copy2(f,dest/f.name)
 checkpoint=checkpoints[-1];j['resume_checkpoint']=str((dest/checkpoint.name).relative_to(root/'demo_project/runs'));j['resume_iteration']=int(checkpoint.stem.split('_')[-1])
 for suffix in ['.log','.exit.json','_guard.log']:
  f=out/(name+suffix)
  if f.exists():shutil.move(str(f),dest/f.name)
save()
for v in plan['variants']:
 name=v['run'];j=state['train'][name]
 if j['status'] not in ('pending','resume_pending'):continue
 while True:
  scan()
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  mem={int(i):(int(t),int(u),int(util)) for i,t,u,util in (row.split(',') for row in rows.splitlines())}
  candidates=[i for i,(t,u,util) in mem.items() if i not in excluded and u+a.reserve_mib<t*.78]
  if candidates:gpu=min(candidates,key=lambda i:(mem[i][1],mem[i][2]));break
  time.sleep(10)
 dst=root/'demo_project/runs'/name
 if subprocess.check_output(['git','-C',str(dst),'rev-parse','HEAD'],text=True).strip()!=plan['commit']:raise RuntimeError('Wrong fixed source')
 if subprocess.check_output(['git','-C',str(dst),'status','--porcelain'],text=True).strip():raise RuntimeError('Dirty fixed source')
 if 'resume_checkpoint' in j:
  initialization=['--checkpoint','/workspace/runs/'+j['resume_checkpoint']];iterations=plan['iterations']-j['resume_iteration']
 else:
  initialization=['--initialize-from','/workspace/runs/'+str(pathlib.Path(v['initial']).relative_to('demo_project/runs'))];iterations=plan['iterations']
 cmd=['python3',str(dst/'tools/run_guarded.py'),'--container',plan['container'],'--gpu',str(gpu),'--reserve-mib',str(a.reserve_mib),'--workdir','/workspace/runs/'+name,'--entrypoint','train_cycle.py','--log',str(out/(name+'.log')),'--headless','--device','cuda:0','--num_envs',str(v['envs']),'--iterations',str(iterations),'--save-interval',str(plan['save_interval']),'--learning-rate',str(plan['learning_rate']),'--seed',str(v['seed']),*initialization,'--log_dir','/workspace/runs/'+name+'/logs/'+name,*j['envelope']]
 with (out/(name+'_guard.log')).open('w') as f:q=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 j.update(pid=q.pid,gpu=gpu,command=cmd,status='initializing');save();deadline=time.monotonic()+900
 while True:
  scan()
  log=out/(name+'.log')
  if log.exists() and 'Learning iteration' in log.read_text(errors='replace'):break
  if time.monotonic()>deadline:raise RuntimeError('Startup timeout: '+name)
  time.sleep(2)
 j['status']='ppo_started';save();print('PPO',name,gpu,flush=True)
scan()
with (out/'eval_queue.log').open('w') as f:q=subprocess.Popen(['python3',str(pathlib.Path(__file__).with_name('evaluate_cycle.py')),'--root',str(root),'--name',plan['name']],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
state.update(eval_pid=q.pid,all_ppo_at=time.strftime('%FT%T%z'));save();print('Evaluation queue',q.pid,flush=True)
while any(j['status']!='complete' for j in state['train'].values()):scan();time.sleep(10)
print('All training exited successfully',flush=True)
