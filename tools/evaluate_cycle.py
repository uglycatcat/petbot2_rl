"""Fixed checkpoints, native envelopes plus common final wide-envelope evaluation."""
import argparse,fcntl,json,pathlib,shutil,subprocess,time
from cycle_evaluation_contract import validate
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--name',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
root=a.root.resolve();out=root/'demo_project/runs'/('_'+a.name)
lock=(out/'eval.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
path=out/'eval_status.json'
if path.exists() and not a.resume:raise RuntimeError('Existing evaluation state: inspect before resuming')
launch=json.loads((out/'launch_status.json').read_text());state={};jobs=[]
eval_reserve=launch.get('eval_reserve_mib',6000)
if eval_reserve<4000:raise ValueError('Evaluation headroom below validated minimum')
if a.resume:
 previous=json.loads(path.read_text())
 if any(v['status']=='running' for v in previous.values()):raise RuntimeError('Resolve running entries before resume')
 for k,v in previous.items():
  if v['status']=='complete':
   validate(v['result']);state[k]=v
 path.write_text(json.dumps(state,indent=2))
for iteration in launch['checkpoints']:
 for name,job in launch['train'].items():jobs.append((name,job,iteration,'native',job['envelope']))
for scope,overrides in launch.get('evaluation_envelopes',{'common':{'--speed-limit':'.1','--yaw-limit':'.3','--frame-min':'-.3','--frame-max':'.3'}}).items():
 for name,job in launch['train'].items():
  envelope=job['envelope'].copy()
  for key,value in overrides.items():
   match=next((i for i,x in enumerate(envelope) if x.startswith(key+'=')),None)
   if match is not None:envelope[match]=key+'='+str(value)
   elif key in envelope:envelope[envelope.index(key)+1]=str(value)
   else:envelope.extend([key,str(value)])
  jobs.append((name,job,launch['iterations']-1,scope,envelope))
if launch.get('heldout_seed') is not None:
 for name,job in launch['train'].items():
  envelope=job['envelope'].copy()
  if launch.get('heldout_hold_seconds'):
   for key in ('--hold-min','--hold-max'):
    if key in envelope:envelope[envelope.index(key)+1]=str(launch['heldout_hold_seconds'])
    else:envelope.extend([key,str(launch['heldout_hold_seconds'])])
  jobs.append((name,job,launch['iterations']-1,'heldout',envelope))
deadline=time.time()+36*3600
while time.time()<deadline:
 pending=False
 for name,job,iteration,scope,envelope in jobs:
  key=f'{name}_i{iteration}_{scope}'
  if state.get(key,{}).get('status') in ('complete','failed','missing_checkpoint'):continue
  pending=True;checkpoint=root/f'demo_project/runs/{name}/logs/{name}/model_{iteration}.pt'
  # Recovery may re-save its starting iteration; retain the pre-interruption checkpoint identity.
  preserved=out/launch.get('recovery_archive','_no_recovery')/name/f'model_{iteration}.pt'
  if preserved.exists():checkpoint=preserved
  if not checkpoint.exists():
   end=out/(name+'.exit.json')
   if end.exists():state[key]=dict(status='missing_checkpoint',training_exit=json.loads(end.read_text()));path.write_text(json.dumps(state,indent=2))
   continue
  if time.time()-checkpoint.stat().st_mtime<30:continue
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
  mem={int(i):(int(t),int(u)) for i,t,u in (x.split(',') for x in rows.splitlines())}
  gpu=next((i for i in (2,3,1) if i not in launch.get('excluded_gpus',[]) and mem[i][1]+eval_reserve<mem[i][0]*.78),None)
  if gpu is None:continue
  state[key]=dict(status='running',gpu=gpu,envelope=envelope,started_at=time.strftime('%FT%T%z'));path.write_text(json.dumps(state,indent=2))
  cmd=['python3',str(pathlib.Path(__file__).with_name('run_guarded.py')),'--container',launch['container'],'--gpu',str(gpu),'--reserve-mib',str(eval_reserve),'--entrypoint','train_cycle.py','--workdir',f'/workspace/runs/{name}','--log',str(out/f'eval_{key}.log'),'--headless','--device','cuda:0','--num_envs','64','--evaluate','--seed',str(launch['heldout_seed'] if scope=='heldout' else (20261019 if scope=='native' else 20261020)),*envelope,'--checkpoint','/workspace/runs/'+str(checkpoint.relative_to(root/'demo_project/runs')),'--log_dir',f'/workspace/runs/_{a.name}/eval_{key}']
  started=time.time();code=subprocess.run(cmd).returncode;result=out/f'eval_{key}/evaluation.json'
  state[key].update(status='complete' if code==0 and result.exists() else 'failed',exit_code=code,finished_at=time.strftime('%FT%T%z'))
  if result.exists():
   try:
    if result.stat().st_mtime < started:raise ValueError('Stale evaluation artifact')
    state[key]['result']=validate(json.loads(result.read_text()))
   except (ValueError,KeyError,TypeError) as error:
    state[key].update(status='failed',validation_error=str(error))
  path.write_text(json.dumps(state,indent=2))
 if not pending:break
 time.sleep(15)
for name,job,iteration,scope,envelope in jobs:
 key=f'{name}_i{iteration}_{scope}'
 if key not in state:state[key]=dict(status='missing_checkpoint')
path.write_text(json.dumps(state,indent=2))
summary=dict(failed=sum(x['status']!='complete' for x in state.values()),finished_at=time.strftime('%FT%T%z'),complete=sum(x['status']=='complete' for x in state.values()),expected=len(jobs))
(out/'eval_finished.json').write_text(json.dumps(summary))
if launch.get('thread') and shutil.which('codex'):
 message=f'upright_rl_demo完整循环批次{a.name}评估队列已结束。先读AGENTS.md、docs/current_handoff.md、docs/upright_stability_first_plan_20261026.md及本批设计/结果文档，核查该批launch_status/eval_status/eval_finished；按需读取docs/upright_training_lineage_20261023.md。先核查实际进程、缺失/失败和结果，再按稳定优先与基本遥控分析优化；旧瞬时精度不作为主要否决条件。用户已授权今晚事件驱动自动迭代，勿恢复周期定时器或goal。不要重复启动；旧移动第五轮及之后夜间结果已废弃，不作优化来源（不指完整循环同号批次）；若受迫中断先继续当前轮；保留最新用户授权与约束，不能把队列结束当任务成功。'
 try:
  r=subprocess.run([shutil.which('codex'),'queue','--thread',launch['thread'],'--message',message],capture_output=True,text=True,timeout=120);notice=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 except Exception as e:notice=dict(submission_unknown=str(e))
 (out/'completion_notice.json').write_text(json.dumps(notice,indent=2))
