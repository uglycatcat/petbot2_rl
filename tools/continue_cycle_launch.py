"""Continue a partial cycle launch without restarting live/checkpointed jobs."""
import argparse,fcntl,json,pathlib,subprocess,time,shutil
p=argparse.ArgumentParser();p.add_argument('plan',type=pathlib.Path);a=p.parse_args();plan=json.loads(a.plan.read_text())
root=pathlib.Path(__file__).resolve().parents[3];out=root/'demo_project/runs'/('_'+plan['name']);statepath=out/'launch_status.json'
lock=(root/'demo_project/runs/upright_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=json.loads(statepath.read_text());assert state['commit']==plan['commit'];assert not state.get('eval_pid')
def save():
 temp=statepath.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(statepath)
def live(pid):
 f=pathlib.Path('/proc')/str(pid)/'cmdline';return f.exists() and b'run_guarded.py' in f.read_bytes()
for v in plan['variants']:
 name=v['run'];j=state['train'].get(name)
 if j and live(j['pid']):
  assert j['status']=='ppo_started', 'Resolve active initialization before continuing'
  continue
 if list((root/'demo_project/runs'/name/'logs'/name).glob('model_*.pt')):raise RuntimeError('Stopped trained job requires checkpoint recovery: '+name)
 if j:
  arc=out/('startup_failure_'+name+'_'+time.strftime('%H%M%S'));arc.mkdir()
  for suffix in ['.log','.exit.json','_guard.log']:
   f=out/(name+suffix)
   if f.exists():shutil.move(str(f),arc/f.name)
 envelope=plan['envelope']+['--wheel-slew',str(v['wheel_slew']),'--frame-slew',str(v['frame_slew'])]
 state['train'][name]=dict(pid=0,gpu=v['gpu'],status='capacity_pending',envelope=envelope,num_envs=v['envs']);save()
 while True:
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  mem={int(i):(int(t),int(u),int(util)) for i,t,u,util in (row.split(',') for row in rows.splitlines())}
  eligible=[i for i,(t,u,util) in mem.items() if u+6000<t*.78]
  if eligible:gpu=min(eligible,key=lambda i:(mem[i][2],mem[i][1]));break
  time.sleep(10)
 dst=root/'demo_project/runs'/name
 assert subprocess.check_output(['git','-C',str(dst),'rev-parse','HEAD'],text=True).strip()==plan['commit']
 initial=pathlib.Path(v['initial']).relative_to('demo_project/runs')
 cmd=['python3',str(dst/'tools/run_guarded.py'),'--container',plan['container'],'--gpu',str(gpu),'--reserve-mib','6000','--workdir','/workspace/runs/'+name,'--entrypoint','train_cycle.py','--log',str(out/(name+'.log')),'--headless','--device','cuda:0','--num_envs',str(v['envs']),'--iterations',str(plan['iterations']),'--save-interval',str(plan['save_interval']),'--learning-rate',str(plan['learning_rate']),'--seed',str(v['seed']),'--initialize-from','/workspace/runs/'+str(initial),'--log_dir','/workspace/runs/'+name+'/logs/'+name,*envelope]
 with (out/(name+'_guard.log')).open('w') as f:q=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 state['train'][name].update(pid=q.pid,gpu=gpu,status='initializing',command=cmd);save()
 deadline=time.monotonic()+900
 while True:
  if q.poll() is not None:raise RuntimeError('Startup failed; inspect before retry: '+name)
  log=out/(name+'.log')
  if log.exists() and 'Learning iteration' in log.read_text(errors='replace'):break
  if time.monotonic()>deadline:raise RuntimeError('Startup timeout: '+name)
  time.sleep(2)
 state['train'][name]['status']='ppo_started';save();print('PPO',name,gpu,flush=True)
with (out/'eval_queue.log').open('w') as f:q=subprocess.Popen(['python3',str(dst/'tools/evaluate_cycle.py'),'--root',str(root),'--name',plan['name']],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
state.update(eval_pid=q.pid,all_ppo_at=time.strftime('%FT%T%z'));save();print('Evaluation queue',q.pid,flush=True)
