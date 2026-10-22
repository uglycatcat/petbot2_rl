"""Fixed checkpoints, native envelopes plus common final wide-envelope evaluation."""
import argparse,fcntl,json,pathlib,shutil,subprocess,time
from evaluation_contract import validate
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
root=a.root.resolve();out=root/'demo_project/runs'/('_'+a.name)
lock=(out/'eval.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
path=out/'eval_status.json'
if path.exists():raise RuntimeError('Existing evaluation state: inspect before resuming')
launch=json.loads((out/'launch_status.json').read_text());state={};jobs=[]
for iteration in launch['checkpoints']:
 for name,job in launch['train'].items():jobs.append((name,job,iteration,'native',job['envelope']))
for name,job in launch['train'].items():
 mode=job['envelope'][job['envelope'].index('--wheel-mode')+1] if '--wheel-mode' in job['envelope'] else 'independent'
 slew=job['envelope'][job['envelope'].index('--wheel-slew')+1] if '--wheel-slew' in job['envelope'] else None
 extras=['--wheel-slew',slew] if slew else []
 if '--feedback-mode' in job['envelope']:extras+=['--feedback-mode',job['envelope'][job['envelope'].index('--feedback-mode')+1]]
 if '--physical-contact-weight' in job['envelope']:extras+=['--physical-contact-weight',job['envelope'][job['envelope'].index('--physical-contact-weight')+1]]
 if '--stability-credit' in job['envelope']:extras+=['--stability-credit',job['envelope'][job['envelope'].index('--stability-credit')+1]]
 jobs.append((name,job,launch['iterations']-1,'common',['--speed-limit','0.3','--yaw-limit','1.0','--frame-min=-1.0','--frame-max','1.2','--wheel-mode',mode,*extras]))
if launch.get('heldout_seed') is not None:
 for name,job in launch['train'].items():jobs.append((name,job,launch['iterations']-1,'heldout',job['envelope']))
deadline=time.time()+36*3600
while time.time()<deadline:
 pending=False
 for name,job,iteration,scope,envelope in jobs:
  key=f'{name}_i{iteration}_{scope}'
  if state.get(key,{}).get('status') in ('complete','failed','missing_checkpoint'):continue
  pending=True;checkpoint=root/f'demo_project/runs/{name}/logs/{name}/model_{iteration}.pt'
  if not checkpoint.exists():
   end=out/(name+'.exit.json')
   if end.exists():state[key]=dict(status='missing_checkpoint',training_exit=json.loads(end.read_text()));path.write_text(json.dumps(state,indent=2))
   continue
  if time.time()-checkpoint.stat().st_mtime<30:continue
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
  mem={int(i):(int(t),int(u)) for i,t,u in (x.split(',') for x in rows.splitlines())}
  gpu=next((i for i in (2,3,1) if mem[i][1]+3500<mem[i][0]*.78),None)
  if gpu is None:continue
  state[key]=dict(status='running',gpu=gpu,envelope=envelope,started_at=time.strftime('%FT%T%z'));path.write_text(json.dumps(state,indent=2))
  cmd=['python3',str(pathlib.Path(__file__).with_name('run_guarded.py')),'--container',launch['container'],'--gpu',str(gpu),'--reserve-mib','3500','--entrypoint','train_moving.py','--workdir',f'/workspace/runs/{name}','--log',str(out/f'eval_{key}.log'),'--headless','--device','cuda:0','--num_envs','64','--evaluate','--seed',str(launch['heldout_seed'] if scope=='heldout' else (20261019 if scope=='native' else 20261020)),*envelope,'--checkpoint',f'/workspace/runs/{name}/logs/{name}/model_{iteration}.pt','--log_dir',f'/workspace/runs/_{a.name}/eval_{key}']
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
 message=f'upright_rl_demo运动任务批次{a.name}评估队列已结束。读取AGENTS.md、docs/upright_repair_20261021.md、对应最新实验文档及该批launch_status/eval_status/eval_finished，先核查实际进程、缺失/失败和结果，再分析优化。不要重复启动；旧第五轮及之后夜间结果已废弃，不作优化来源；若受迫中断先继续当前轮；保留最新用户授权与约束，不能把队列结束当任务成功。'
 try:
  r=subprocess.run([shutil.which('codex'),'queue','--thread',launch['thread'],'--message',message],capture_output=True,text=True,timeout=120);notice=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 except Exception as e:notice=dict(submission_unknown=str(e))
 (out/'completion_notice.json').write_text(json.dumps(notice,indent=2))
