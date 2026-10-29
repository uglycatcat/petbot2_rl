"""Finite, guarded random-initialization jobs and fixed-policy evaluations.

Run on the host. Every simulation runs inside the explicitly named container.
No historical model path is accepted. A failed step stops that run; an event
asks the agent to inspect evidence and recover the same run if appropriate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

SOURCE=Path(__file__).resolve().parents[1]
ROOT=SOURCE.parents[1]
RUNS=ROOT/'demo_project/runs'
mutex=threading.Lock()

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def git(*args):return subprocess.check_output(['git','-C',str(SOURCE),*args],text=True).strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan',type=Path)
    a=parser.parse_args();plan=json.loads(a.plan.read_text())
    if plan['mode'] not in ('engineering','formal'):raise ValueError('Unknown batch mode')
    if plan.get('pretrained_models')!=[]:raise ValueError('No pretrained model is allowed')
    commit=git('rev-parse','HEAD')
    if git('status','--porcelain') or commit!=plan['commit']:raise RuntimeError('Source must be clean and pinned')
    lock=(RUNS/'upright_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    variants=plan['variants']
    assert len({v['name'] for v in variants})==len(variants)
    assert len({v['gpu'] for v in variants})==len(variants), 'One process per selected GPU'
    for v in variants:
        assert v['gpu'] in (1,2,3) and 1<=v['num_envs']<=4096
        assert not (RUNS/v['name']).exists(), 'Run already exists; inspect/recover it'
        if plan['mode']=='formal':assert v['iterations']==1000
    if plan['mode']=='formal':
        gate=json.loads((ROOT/plan['gate']).read_text())
        if not gate['passed']:raise RuntimeError('Engineering gate did not pass')
        if git('diff',gate['tested_commit'],commit,'--','train_cycle.py','train_scratch.py','upright','configs'):
            raise RuntimeError('Training implementation changed since engineering acceptance')
    out=RUNS/('_'+plan['name']);out.mkdir(exist_ok=False)
    save(out/'plan.json',plan)
    state=dict(name=plan['name'],mode=plan['mode'],commit=commit,pretrained_models=[],
        eval_pid=os.getpid(),evaluator_entrypoint='launch_scratch.py',
        thread=os.environ.get('CODEX_THREAD_ID'),started_at=time.strftime('%FT%T%z'),train={})
    status={}
    save(out/'launch_status.json',state);save(out/'eval_status.json',status)
    with (out/'eval_watch.log').open('x') as log:
        subprocess.Popen(['python3',str(SOURCE/'tools/watch_cycle_evaluation.py'),str(out)],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

    def step(v,work,name,entry,args):
        key=v['name']+'_'+name;log=out/(key+'.log')
        command=['python3',str(work/'tools/run_guarded.py'),'--container',plan['container'],
            '--gpu',str(v['gpu']),'--reserve-mib',str(plan.get('reserve_mib',8000)),
            '--workdir','/workspace/runs/'+v['name'],'--entrypoint',entry,'--log',str(log),*args]
        with (out/(key+'_guard.log')).open('x') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            with mutex:
                status[key]=dict(status='running',pid=child.pid,command=command)
                save(out/'eval_status.json',status)
            code=child.wait()
        exit_path=log.with_suffix('.exit.json')
        evidence=json.loads(exit_path.read_text()) if exit_path.exists() else {'missing_exit':True}
        okay=code==0 and evidence.get('exit_code')==0 and not any(evidence.get(k) for k in ('guard_stopped','physics_error','runtime_error'))
        with mutex:
            status[key].update(status='complete' if okay else 'failed',exit=evidence)
            save(out/'eval_status.json',status)
        if not okay:raise RuntimeError('Step failed: '+key)

    def run(v):
        work=RUNS/v['name']
        try:
            subprocess.run(['git','clone','--quiet','--no-local',str(SOURCE),str(work)],check=True,
                env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'})
            subprocess.run(['git','-C',str(work),'checkout','--quiet','--detach',commit],check=True)
            with mutex:
                state['train'][v['name']]=dict(gpu=v['gpu'],seed=v['seed'],status='preparing',initialization='random')
                save(out/'launch_status.json',state)
            # Deliberately rebuild from read-only MJCF, without an old asset cache.
            step(v,work,'asset','tools/prepare_asset.py',['--headless','--device','cuda:0'])
            args=['--run-dir','logs/scratch','--seed',str(v['seed']),'--iterations',str(v['iterations']),
                '--num_envs',str(v['num_envs']),'--save-interval',str(v.get('save_interval',100)),
                '--headless','--device','cuda:0']
            if v.get('stop_after'):args+=['--stop-after',str(v['stop_after'])]
            step(v,work,'train','train_scratch.py',args)
            if v.get('stop_after'):
                point=f"logs/scratch/model_{v['stop_after']-1}.pt"
                step(v,work,'resume','train_scratch.py',['--run-dir','logs/scratch','--resume',point,'--headless','--device','cuda:0'])
            checkpoint=f"logs/scratch/model_{v['iterations']-1}.pt"
            if not (work/checkpoint).is_file():raise RuntimeError('Final checkpoint missing')
            with mutex:
                state['train'][v['name']]['status']='trained';save(out/'launch_status.json',state)
            no_balance=False
            for scenario in plan['evaluations']:
                if no_balance:
                    with mutex:
                        status[v['name']+'_eval_'+scenario['name']]=dict(status='skipped',
                            reason='Basic evaluation reached no two-wheel state; analyze core learning before extending evaluation')
                        save(out/'eval_status.json',status)
                    continue
                args=['--run-dir','logs/scratch','--evaluate',checkpoint,'--output','logs/eval_'+scenario['name'],
                    '--num_envs','64','--headless','--device','cuda:0',*scenario['args']]
                step(v,work,'eval_'+scenario['name'],'train_scratch.py',args)
                result=json.loads((work/('logs/eval_'+scenario['name']+'/evaluation.json')).read_text())
                no_balance=plan.get('skip_remaining_on_no_balance',False) and result['balance']==0
            return dict(name=v['name'],execution_complete=True,acceptance_pending=True)
        except Exception as error:
            with mutex:
                state['train'].setdefault(v['name'],{}).update(status='failed',error=repr(error))
                save(out/'launch_status.json',state)
            return dict(name=v['name'],execution_complete=False,error=repr(error),acceptance_pending=True)

    with ThreadPoolExecutor(max_workers=len(variants)) as pool:results=list(pool.map(run,variants))
    save(out/'eval_finished.json',dict(results=results,finished_at=time.strftime('%FT%T%z'),acceptance_pending=True))
    message=(f"upright_rl_demo从零训练队列{plan['name']}已结束。读取AGENTS.md、docs/current_handoff.md、"
        "docs/upright_cycle_v26.md和该批状态、实际进程、失败/显存中断、随机初始化及同次恢复证据、"
        "请求/执行/实测轨迹。先验收分析再继续已授权事件迭代；禁止加载历史父模型，工程退出0不代表策略通过。"
        "保留C58体验基准、双向连续0–0.75m/s、起立少调整、落地只增默认髋位、每份1000更新要求。"
        "不重复启动，不恢复周期定时器或goal。")
    executable=shutil.which('codex')
    if executable and state['thread']:
        save(out/'completion_notice.json',dict(submission_unknown='attempt_started'))
        try:
            r=subprocess.run([executable,'queue','--thread',state['thread'],'--message',message],capture_output=True,text=True,timeout=120)
            notice=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except Exception as error:notice=dict(submission_unknown=repr(error))
        save(out/'completion_notice.json',notice)
    print(json.dumps(results))

if __name__=='__main__':main()
