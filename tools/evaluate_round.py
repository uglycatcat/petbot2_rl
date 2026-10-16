"""Evaluate a fixed, explicitly launched four-GPU round; never starts training."""
import argparse,fcntl,json,os,pathlib,shutil,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--round',required=True);a=p.parse_args()
root=a.root.resolve();out=root/'demo_project/runs'/('_upright_'+a.round)
lock=(out/'eval.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
path=out/'eval_status.json'
if path.exists():raise RuntimeError('Existing evaluator state: inspect before restart')
launch=json.loads((out/'launch_status.json').read_text());state={}
iterations=launch['checkpoints'];jobs=launch['train'];deadline=time.time()+36*3600
while time.time()<deadline:
 pending=False
 for iteration in iterations:
  for name,job in jobs.items():
   key=f'{name}_i{iteration}'
   if state.get(key,{}).get('status') in ('complete','failed','missing_checkpoint'):continue
   pending=True;checkpoint=root/f'demo_project/runs/{name}/logs/{name}/model_{iteration}.pt'
   if not checkpoint.exists():
    end=out/f'{name}.exit.json'
    if end.exists():state[key]={'status':'missing_checkpoint','training_exit':json.loads(end.read_text())};path.write_text(json.dumps(state,indent=2))
    continue
   if time.time()-checkpoint.stat().st_mtime<30:continue
   # Evaluation goes on GPU2/3/1 in that order, only when total memory has room.
   rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
   mem={int(i):(int(t),int(u)) for i,t,u in (line.split(',') for line in rows.splitlines())}
   gpu=next((g for g in (2,3,1) if mem[g][1]+3500<mem[g][0]*.78),None)
   if gpu is None:continue
   state[key]={'status':'running','gpu':gpu,'started_at':time.strftime('%FT%T%z')};path.write_text(json.dumps(state,indent=2))
   command=['python3',str(pathlib.Path(__file__).with_name('run_guarded.py')),'--container',launch['container'],'--gpu',str(gpu),'--reserve-mib','3500',
    '--workdir',f'/workspace/runs/{name}','--log',str(out/f'eval_{key}.log'),'--headless','--device','cuda:0','--num_envs','64',
    '--evaluate','--seed','20261016','--beta-shape-cap',str(job.get('shape_cap',32.)),'--settle-weight',str(job.get('settle_weight',0.)),'--control-decimation',str(job['decimation']),'--checkpoint',f'/workspace/runs/{name}/logs/{name}/model_{iteration}.pt',
    '--log_dir',f'/workspace/runs/_upright_{a.round}/eval_{key}']
   rc=subprocess.run(command).returncode;result=out/f'eval_{key}/evaluation.json'
   state[key].update(status='complete' if rc==0 and result.exists() else 'failed',exit_code=rc,finished_at=time.strftime('%FT%T%z'))
   if result.exists():
    state[key]['result']=json.loads(result.read_text())
    with (out/f'render_{key}.log').open('w') as f:
     rr=subprocess.run(['artifacts/mujoco_env/bin/python','robot_controller_demo/jump_demo/render_upright_trace.py',str(result.parent)],cwd=root,
      env={**os.environ,'MUJOCO_GL':'egl','MUJOCO_EGL_DEVICE_ID':str(gpu),'MPLCONFIGDIR':str(root/'artifacts/matplotlib')},stdout=f,stderr=subprocess.STDOUT)
    state[key]['render_exit_code']=rr.returncode
   path.write_text(json.dumps(state,indent=2))
 if not pending:break
 time.sleep(15)
(out/'eval_finished.json').write_text(json.dumps({'finished_at':time.strftime('%FT%T%z'),'complete':sum(v['status']=='complete' for v in state.values()),'expected':len(jobs)*len(iterations)}))

# Report completion to the same authorized session so analysis need not wait for a timer.
if launch.get('thread') and shutil.which('codex'):
 message=('upright_rl_demo 自动迭代轮次 '+a.round+' 的评估队列已结束。读取 docs/tonight_training_20261016.md、该轮 launch_status.json/eval_status.json/eval_finished.json，核查实际进程与结果；先分析再决定代码修改与下一轮，禁止直接重复启动。遵守最多4轮、2026-10-17 08:00之后完成当时所在轮便收官的边界。失败或缺失结果先诊断，不冒称完整评估通过。')
 try:
  r=subprocess.run([shutil.which('codex'),'queue','--thread',launch['thread'],'--message',message],capture_output=True,text=True,timeout=120)
  notice={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 except Exception as e:notice={'submission_unknown':str(e)}
 (out/'completion_notice.json').write_text(json.dumps(notice,indent=2))
