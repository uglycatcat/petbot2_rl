"""Event-driven project campaign: immutable inputs, scratch PPO, evaluation, next-round choice.
No cron/system timer and no access to other projects' controls or processes.
"""
from pathlib import Path
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time

PROJECT=Path(__file__).resolve().parents[1]
CONTAINER_ROOT=Path('/workspace/petbot/demo_project/step_explore')
CONTROL=PROJECT/'control'

def atomic(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)

def settings():return json.loads((CONTROL/'control.json').read_text())

def container_path(path):return str(CONTAINER_ROOT/path.resolve().relative_to(PROJECT))

def free_mib(gpu):
    return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits',f'--id={gpu}'],text=True).strip())

def run_job(gpu,batch,config):
    out=batch/f'gpu{gpu}'
    out.mkdir()
    atomic(out/'config.json',config)
    if free_mib(gpu)<6500:
        return {'gpu':gpu,'status':'resource_wait','reason':'less_than6500MiB_free'}
    if not settings()['enabled']:
        return {'gpu':gpu,'status':'cancelled_before_start'}
    cmd=['docker','exec','-e','PYTHONUNBUFFERED=1','-w',container_path(batch/'source'),
        f'petbot-step-explore-gpu{gpu}','/workspace/isaaclab/isaaclab.sh','-p','train.py',
        '--config',container_path(out/'config.json'),'--output',container_path(out),
        '--asset-dir',container_path(batch/'assets'),'--headless','--device','cuda:0']
    with (out/'console.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        atomic(out/'process.json',{'pid':proc.pid,'gpu':gpu,'command':cmd,'started_at':time.time()})
        # Wait on the child handle while recording this job's actual GPU memory.
        stop_sent=False
        while True:
            try:
                code=proc.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                free=free_mib(gpu)
                top=subprocess.check_output(['docker','top',f'petbot-step-explore-gpu{gpu}','-eo','pid,args'],text=True)
                pids={line.split()[0] for line in top.splitlines()[1:] if 'python3 train.py' in line}
                usage=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                    '--format=csv,noheader,nounits',f'--id={gpu}'],text=True)
                own=sum(int(line.split(',')[1].strip()) for line in usage.splitlines()
                    if line.split(',')[0].strip() in pids)
                with (out/'resources.jsonl').open('a') as f:
                    f.write(json.dumps({'time':time.time(),'card_free_mib':free,'own_compute_mib':own})+'\n')
                if not stop_sent and (free<1024 or settings().get('stop_current',False)):
                    pattern='python3 train.py.*'+re.escape(container_path(out))+'( |$)'
                    subprocess.run(['docker','exec',f'petbot-step-explore-gpu{gpu}','pkill','-USR1','-f',pattern],check=False)
                    stop_sent=True
                    atomic(out/'stop_request.json',{'reason':'memory_pressure' if free<1024 else 'user_control',
                        'free_mib':free,'requested_at':time.time()})
    status_path=out/'status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('status')=='interrupted':
        return {'gpu':gpu,'status':'interrupted','detail':json.loads(status_path.read_text())}
    error=out/'error.json'
    if error.exists():return {'gpu':gpu,'status':'error','detail':json.loads(error.read_text())}
    checkpoint=out/'final.pt'
    if not code and checkpoint.exists():
        ev=out/'eval'
        eval_cmd=cmd.copy()
        oi=eval_cmd.index('--output');eval_cmd[oi+1]=container_path(ev)
        eval_cmd+=['--mode','evaluate','--checkpoint',container_path(checkpoint),
                   '--num-envs',str(config['eval_episodes'])]
        with (out/'evaluate.log').open('w') as log:
            evaluated=subprocess.Popen(eval_cmd,stdout=log,stderr=subprocess.STDOUT)
            atomic(out/'eval_process.json',{'pid':evaluated.pid,'gpu':gpu,'started_at':time.time()})
            code=evaluated.wait()
        if (ev/'error.json').exists():return {'gpu':gpu,'status':'evaluation_error','detail':json.loads((ev/'error.json').read_text())}
    result=out/'eval'/'evaluation.json'
    if code or not result.exists() or not checkpoint.exists():
        return {'gpu':gpu,'status':'error','return_code':code,'reason':'missing_verified_outputs'}
    reviewed=subprocess.run(['docker','exec',f'petbot-step-explore-gpu{gpu}',
        '/isaac-sim/python.sh',container_path(batch/'source/tools/review.py'),'--run',container_path(out)],
        capture_output=True,text=True)
    (out/'review.log').write_text(reviewed.stdout+reviewed.stderr)
    if reviewed.returncode or not (out/'trajectory_review.json').exists():
        return {'gpu':gpu,'status':'review_error','return_code':reviewed.returncode}
    return {'gpu':gpu,'status':'evaluated','evaluation':json.loads(result.read_text()),
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}

def main():
    CONTROL.mkdir(exist_ok=True)
    lock=(CONTROL/'campaign.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(CONTROL/'process.json',{'pid':os.getpid(),'started_at':time.time()})
    cfg=settings()
    round_number=cfg.get('next_round',1)
    previous=[]
    while settings()['enabled']:
        if subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT,text=True).strip():
            raise RuntimeError('Publish a clean project commit before freezing a batch')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
        remote_name=subprocess.check_output(['git','config','--get','branch.demoproj/step_explore.remote'],cwd=PROJECT,text=True).strip()
        remote=subprocess.check_output(['git','ls-remote',remote_name,'refs/heads/demoproj/step_explore'],cwd=PROJECT,text=True).split()[0]
        if remote!=commit:raise RuntimeError('Local source has not been published')
        batch=PROJECT/'runs'/f'round{round_number:03d}_{time.strftime("%Y%m%d_%H%M%S")}'
        batch.mkdir(parents=True)
        (batch/'source').mkdir();(batch/'assets').mkdir()
        archive=batch/'source.tar'
        with archive.open('wb') as f:subprocess.run(['git','archive',commit],cwd=PROJECT,stdout=f,check=True)
        with tarfile.open(archive) as t:t.extractall(batch/'source',filter='data')
        for n in ['robot.usda','model.json']:shutil.copy2(PROJECT/'assets'/n,batch/'assets'/n)
        baseline=json.loads((PROJECT/'configs/round01.json').read_text())
        jobs={}
        for gpu in [0,1]:
            c=dict(baseline)
            c['seed']=100+round_number*2+gpu
            # Prespecified small exploration comparison; task and acceptance never change.
            c['initial_std']=[.65,.9,.45][(round_number-1+gpu)%3]
            c['entropy_coef']=[.01,.02,.005][(round_number-1+gpu)%3]
            if previous and max([r.get('evaluation',{}).get('successes',0) for r in previous],default=0)==0:
                c['exploration_bonus']=[10.,15.,7.][(round_number-1)%3]
            jobs[gpu]=c
        atomic(batch/'batch.json',{'commit':commit,'round':round_number,'jobs':jobs,'from_scratch':True,
            'previous_review':previous,'created_at':time.time()})
        atomic(CONTROL/'active.json',{'batch':str(batch.relative_to(PROJECT)),'status':'training','commit':commit})
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run_job,g,batch,c) for g,c in jobs.items()]
            results=[f.result() for f in futures]
        atomic(batch/'review.json',{'results':results,'reviewed_at':time.time(),
            'interpretation':'Independent recorded geometry and four-wheel top contacts; full trajectory review still required for delivery.'})
        if any(r['status']!='evaluated' for r in results):
            atomic(CONTROL/'active.json',{'batch':str(batch.relative_to(PROJECT)),'status':'needs_engineering_review','results':results})
            return
        if max(r['evaluation']['success_rate'] for r in results)>=.9:
            atomic(CONTROL/'active.json',{'batch':str(batch.relative_to(PROJECT)),'status':'candidate_for_delivery_review','results':results})
            return
        previous=results;round_number+=1
        cfg=settings();cfg['next_round']=round_number;atomic(CONTROL/'control.json',cfg)
        atomic(CONTROL/'active.json',{'batch':str(batch.relative_to(PROJECT)),'status':'reviewed_next_scratch_batch','results':results})

if __name__=='__main__':main()
