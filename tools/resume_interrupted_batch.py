"""Explicit recovery of a fully stopped batch; preserves source and optimizer state."""
import argparse,json,pathlib,subprocess,shutil,time,fcntl
p=argparse.ArgumentParser();p.add_argument('batch');a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[3];out=root/'demo_project/runs'/('_'+a.batch)
lock=(out/'recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
l=json.loads((out/'launch_status.json').read_text())
for j in [*l['train'].values(),{'pid':l.get('eval_pid',0)}]:
 f=pathlib.Path('/proc')/str(j['pid'])/'cmdline'
 if f.exists() and f.read_bytes():raise RuntimeError('Prior PID still exists; inspect before resuming')
archive=out/'interruption_20261022_1126';archive.mkdir(exist_ok=False)
for name in ['launch_status.json','eval_status.json','eval_finished.json','completion_notice.json']:
 f=out/name
 if f.exists():shutil.copy2(f,archive/name)
for name in ['eval_finished.json','completion_notice.json']:(out/name).unlink(missing_ok=True)
reservations={i:0 for i in range(4)}
for j in l['train'].values():reservations[j['gpu']]+=3600
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
for row in rows.splitlines():
 i,t,u=map(int,row.split(','));assert u+reservations[i]<t*.78,(i,u,reservations[i])
for name,j in l['train'].items():
 run=root/'demo_project/runs'/name;folder=run/'logs'/name
 assert subprocess.check_output(['git','-C',str(run),'rev-parse','HEAD'],text=True).strip()==l['commit']
 assert not subprocess.check_output(['git','-C',str(run),'status','--porcelain'],text=True).strip()
 checkpoints=list(folder.glob('model_*.pt'));ck=max(checkpoints,key=lambda x:int(x.stem.split('_')[1]));iteration=int(ck.stem.split('_')[1]);assert iteration<599
 saved=archive/name;saved.mkdir();shutil.copy2(ck,saved/ck.name);shutil.copy2(folder/'config.json',saved/'config.json')
 for suffix in ['.log','.exit.json','_guard.log']:
  f=out/(name+suffix)
  if f.exists():f.rename(archive/f.name)
 c=j['command'].copy();c[1]=str(pathlib.Path(__file__).with_name('run_guarded.py'));idx=c.index('--initialize-from');c[idx]='--checkpoint';c[idx+1]='/workspace/runs/'+name+'/logs/'+name+'/'+ck.name
 c[c.index('--iterations')+1]=str(l['iterations']-iteration)
 with (out/(name+'_guard.log')).open('w') as f:q=subprocess.Popen(c,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 j.update(pid=q.pid,command=c,status='resuming',resume_checkpoint=ck.name,resume_started_at=time.strftime('%FT%T%z'))
 (out/'launch_status.json').write_text(json.dumps(l,indent=2))
# The queue uses repaired orchestration, with original fixed task checkouts.
with (out/'eval_queue_resume.log').open('w') as f:q=subprocess.Popen(['python3',str(pathlib.Path(__file__).with_name('evaluate_moving.py')),'--root',str(root),'--name',a.batch,'--resume'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
l.update(eval_pid=q.pid,recovery_orchestration_commit=subprocess.check_output(['git','-C',str(pathlib.Path(__file__).parents[1]),'rev-parse','HEAD'],text=True).strip())
(out/'launch_status.json').write_text(json.dumps(l,indent=2));print('Recovered launch; verify actual PPO and queue',q.pid)
