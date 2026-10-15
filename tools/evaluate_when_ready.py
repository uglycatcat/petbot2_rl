"""One-shot evaluation queue for the first upright batch; no retraining or automatic resume."""
import fcntl,json,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parents[3]
out=root/'demo_project/runs/_night_20261015';out.mkdir(exist_ok=True)
lock=(out/'upright_eval.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state_path=out/'upright_eval_status.json'
state=json.loads(state_path.read_text()) if state_path.exists() else {}
for _ in range(24*360):
    remaining=False
    for iteration in (250,750,1499):
        for seed in (42,43):
            name=f'upright_v1_s{seed}_i{iteration}'
            if state.get(name,{}).get('status') in ('complete','failed'):continue
            remaining=True
            run=f'upright_v1_s{seed}'
            checkpoint=root/f'demo_project/runs/{run}/logs/{run}/model_{iteration}.pt'
            if not checkpoint.exists() or time.time()-checkpoint.stat().st_mtime<15:continue
            row=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
            total,used=map(int,row.split(','))
            if used+3500>total*.78:continue
            state[name]={'status':'running','checkpoint':str(checkpoint.relative_to(root))};state_path.write_text(json.dumps(state,indent=2))
            command=['python3',str(pathlib.Path(__file__).with_name('run_guarded.py')),'--gpu','1','--reserve-mib','3500',
                     '--workdir',f'/workspace/runs/{run}','--log',str(out/f'eval_{name}.log'),'--headless','--device','cuda:0',
                     '--num_envs','64','--evaluate','--seed','20261015','--checkpoint',f'/workspace/runs/{run}/logs/{run}/model_{iteration}.pt',
                     '--log_dir',f'/workspace/runs/_night_20261015/eval_{name}']
            rc=subprocess.run(command).returncode
            result=out/f'eval_{name}/evaluation.json'
            state[name].update(status='complete' if rc==0 and result.exists() else 'failed',exit_code=rc)
            if result.exists():state[name]['result']=json.loads(result.read_text())
            state_path.write_text(json.dumps(state,indent=2))
    if not remaining:break
    time.sleep(10)
