"""Resume an explicitly listed set of stopped jobs sequentially with conservative reload headroom."""
import argparse,fcntl,json,pathlib,subprocess,time,shutil,os,signal
p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('names',nargs='+');a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[3];out=root/'demo_project/runs'/('_'+a.batch)
lock=(out/'recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
l=json.loads((out/'launch_status.json').read_text())
# Caller has already stopped queue scheduler; let any current evaluation finish.
state=json.loads((out/'eval_status.json').read_text())
for key,v in state.items():
 if v['status']=='running':
  f=out/('eval_'+key+'.exit.json');deadline=time.monotonic()+1200
  while not f.exists():
   if time.monotonic()>deadline:raise RuntimeError('Evaluation did not exit')
   time.sleep(2)
  v.update(status='failed',validation_error='Scheduler interrupted; rerun with fresh validation')
(out/'eval_status.json').write_text(json.dumps(state,indent=2))
for name,j in l['train'].items():
 if j['status']=='reload_pending':
  deadline=time.monotonic()+900
  while 'Learning iteration' not in (out/(name+'.log')).read_text(errors='replace'):
   if (out/(name+'.exit.json')).exists() or time.monotonic()>deadline:raise RuntimeError('Prior reload failed')
   time.sleep(2)
  j['status']='ppo_resumed'
(out/'launch_status.json').write_text(json.dumps(l,indent=2))
for name in a.names:
 j=l['train'][name];proc=pathlib.Path('/proc')/str(j['pid'])/'cmdline'
 if proc.exists() and proc.read_bytes():raise RuntimeError('Job still live: '+name)
 # 6 GB includes checkpoint/optimizer reload peak. Recheck at each launch.
 while True:
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
  mem={int(i):(int(t),int(u)) for i,t,u in (r.split(',') for r in rows.splitlines())}
  if mem[a.gpu][1]+6000<mem[a.gpu][0]*.78:break
  time.sleep(10)
 arc=out/('guard_retry_'+name+'_'+time.strftime('%H%M%S'));arc.mkdir()
 for suffix in ['.log','.exit.json','_guard.log']:
  f=out/(name+suffix)
  if f.exists():f.rename(arc/f.name)
 c=j['command'].copy();c[1]=str(pathlib.Path(__file__).with_name('run_guarded.py'));c[c.index('--gpu')+1]=str(a.gpu);c[c.index('--reserve-mib')+1]='6000'
 with (out/(name+'_guard.log')).open('w') as f:q=subprocess.Popen(c,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 j.update(pid=q.pid,gpu=a.gpu,command=c,status='reload_pending');(out/'launch_status.json').write_text(json.dumps(l,indent=2))
 deadline=time.monotonic()+900
 while True:
  if (out/(name+'.exit.json')).exists():raise RuntimeError('Reload failed: '+name)
  log=out/(name+'.log')
  if log.exists() and 'Learning iteration' in log.read_text(errors='replace'):break
  if time.monotonic()>deadline:raise RuntimeError('Reload timeout')
  time.sleep(2)
 j['status']='ppo_resumed';(out/'launch_status.json').write_text(json.dumps(l,indent=2));print('resumed',name,flush=True)
# No competing evaluation allocations during reloads; afterwards restore queue.
with (out/'eval_queue_conservative.log').open('w') as f:q=subprocess.Popen(['python3',str(pathlib.Path(__file__).with_name('evaluate_moving.py')),'--root',str(root),'--name',a.batch,'--resume'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
l['eval_pid']=q.pid;(out/'launch_status.json').write_text(json.dumps(l,indent=2));print('evaluation resumed',q.pid,flush=True)
