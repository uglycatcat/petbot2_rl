"""One-shot acceptance and launch; fail closed, never restart training."""
import fcntl, json, math, os, pathlib, subprocess, time
ROOT=next(p for p in (pathlib.Path.cwd(),*pathlib.Path.cwd().parents) if (p/'default.xml').exists());RUNS=ROOT/'demo_project/runs'
OUT=RUNS/'_sweep4_precheck';OUT.mkdir(exist_ok=True)
NIGHT=RUNS/'_night_20261015';SOURCE=ROOT/'demo_project/recovery_agile_01'
lock=(OUT/'accept.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
status={'status':'waiting_short_training','tested_commit':commit}
def save(): (OUT/'accept_status.json').write_text(json.dumps(status,indent=2))
save()
try:
    paths={}
    deadline=time.time()+3600
    while time.time()<deadline:
        for k in ['b','c']:
            candidates=list((RUNS/f'sweep4_fixed_{k}').glob('logs/rsl_rl/recovery_agile_01/*/model_99.pt'))
            log=(NIGHT/f'sweep4_fixed_{k}.log').read_text()
            if len(candidates)==1 and '[INFO]: SimulationContext cleared' in log and 'Traceback (most recent call last)' not in log:
                paths[k]=candidates[0]
        if len(paths)==2:break
        time.sleep(10)
    if len(paths)!=2:raise RuntimeError('Short training did not finish normally by deadline')
    status['status']='checking_metrics';save()
    # TensorBoard is available in the training container, not assumed on the host.
    probe='''import json,math
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
r={}
for k in ['b','c']:
 p=list(Path('/workspace/runs').glob(f'sweep4_fixed_{k}/logs/rsl_rl/recovery_agile_01/*/model_99.pt'))[0]
 e=EventAccumulator(str(p.parent),size_guidance={'scalars':0});e.Reload();tags=e.Tags()['scalars']
 assert all(math.isfinite(v.value) for t in tags for v in e.Scalars(t)), 'nonfinite metric'
 m={t:{'last':e.Scalars(t)[-1].step,'max':max(v.value for v in e.Scalars(t)),'last20':sum(v.value for v in e.Scalars(t)[-20:])/len(e.Scalars(t)[-20:])} for t in tags}
 for t in ['Episode_Reward/recovery_region','Episode_Reward/low_speed','Episode_Reward/success']:
  assert m[t]['last']==99 and m[t]['last20']>0, (k,t,m[t])
 assert m['Loss/learning_rate']['max']<=.00020001
 r[k]={'all_finite':True,'metrics':m,'provenance':json.loads((p.parent/'params/petbot.json').read_text())}
print(json.dumps(r))'''
    result=subprocess.check_output(['docker','exec','petbot-train','/isaac-sim/python.sh','-c',probe],text=True)
    checks=json.loads(result);(OUT/'short_metrics.json').write_text(json.dumps(checks,indent=2))
    runtime=json.loads((NIGHT/'sweep4_runtime.json').read_text());assert runtime['passed']
    assert '1 passed' in (NIGHT/'sweep4_probability.log').read_text()
    # Physical/probability implementation unchanged since their tested commit; only PPO LR config changed.
    diff=subprocess.check_output(['git','-C',str(SOURCE),'diff',runtime['demo_commit'],commit,'--','agile/petbot/distribution.py','agile/petbot/mdp.py','agile/petbot/core.py','agile/petbot/env_cfg.py','agile/petbot/asset_cfg.py'],text=True)
    assert not diff, 'Physical/probability implementation changed'
    for k in ['b','c']:
        assert checks[k]['provenance']['demo_commit']==commit
        if k=='c':assert checks[k]['provenance']['easy_curriculum']['count']>0
        status['status']=f'reloading_{k}';save()
        path='/workspace/runs/'+str(paths[k].relative_to(RUNS))
        command=['docker','exec','-w',f'/workspace/runs/sweep4_fixed_{k}','-e',f'PYTHONPATH=/workspace/runs/sweep4_fixed_{k}',
                 '-e','CUDA_VISIBLE_DEVICES=3','-e','OMP_NUM_THREADS=2','petbot-train','/isaac-sim/python.sh','scripts/petbot_sweep2_eval.py',
                 '--headless','--episodes','512','--mode','policy','--checkpoint',path,'--dataset','/workspace/runs/_sweep3/eval_states.pt',
                 '--allow-legacy-xyzw-dataset','--output',f'/workspace/runs/_sweep4_precheck/reload_{k}.json']
        with (OUT/f'reload_{k}.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
        d=json.loads((OUT/f'reload_{k}.json').read_text());baseline=json.loads((RUNS/'_sweep3/default.json').read_text())
        assert d['dataset_sha256']==baseline['dataset_sha256']
        assert d['summary']['all']['count']==512 and d['summary']['all']['success_rate']>0
        assert d['action_stats']['clipped']==0, 'Deterministic actions reach/exceed action limits'
    gate={'passed':True,'tested_commit':commit,'physical_commit':runtime['demo_commit'],'checks':'short_metrics.json',
          'reloads':['reload_b.json','reload_c.json'],'note':'positive feedback, finite PPO, fixed LR and reload; not a learned recovery effectiveness claim'}
    (OUT/'gate.json').write_text(json.dumps(gate,indent=2))
    # No overlapping old experiment; preserve its final evaluation before the next launch.
    old=json.loads((RUNS/'_sweep3/status.json').read_text());assert old.get('finished_at')
    assert all(v['status']=='complete' for v in [*old['train'].values(),*old['eval'].values()])
    status['status']='launching';save()
    with (RUNS/'_sweep4_pipeline.log').open('w') as f:
        proc=subprocess.Popen(['python3',str(ROOT/'demo_project/demo_env/sweep4_pipeline.py'),'--root',str(ROOT)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    status.update(status='pipeline_started',pipeline_pid=proc.pid);save()
except BaseException as exc:
    status.update(status='failed',error=repr(exc));save();raise
