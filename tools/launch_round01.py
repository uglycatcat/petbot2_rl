"""Finite guarded scratch training + minimal evaluations; analysis continues via one-shot events."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from snapshot import SOURCE, REPOSITORY, RUNS, create
from events import save, notify
sys.path.insert(0,str(SOURCE))
from upright.model import task_implementation_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('plan',type=Path)
    args=parser.parse_args(); plan=json.loads(args.plan.read_text())
    if plan['initialization']!='random' or plan['historical_models']!=[]:
        raise ValueError('Only independent scratch trials are permitted')
    if sorted(x['gpu'] for x in plan['variants']) != [2,3]:
        raise ValueError('This round uses exactly physical GPU2 and GPU3')
    commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if commit!=plan['commit'] or subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Require the clean fixed commit in the plan')
    gate=json.loads((REPOSITORY/plan['engineering_gate']).read_text())
    if not gate['accepted'] or gate['code_sha256']!=task_implementation_fingerprint():
        raise RuntimeError('Engineering gate missing or implementation changed')
    authorization=json.loads((SOURCE/'project.json').read_text())
    if not authorization['training_enabled'] or not authorization['automatic_iteration_enabled']:
        raise RuntimeError('Automatic training is not enabled')
    thread=os.environ.get('CODEX_THREAD_ID')
    if not thread or not shutil.which('codex'): raise RuntimeError('Event continuation unavailable')
    lock=(RUNS/'upright_restart_launch.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    name=plan['name']
    if Path(name).name!=name: raise ValueError('Invalid round name')
    out=RUNS/('_'+name); out.mkdir(exist_ok=False)
    state=dict(name=name,pid=os.getpid(),thread=thread,commit=commit,allowed_host_gpus=[2,3],
               started_at=time.time(),variants={})
    status={}; mutex=threading.Lock()
    save(out/'plan.json',plan); save(out/'launch_status.json',state); save(out/'eval_status.json',status)
    with (out/'exit_watch.log').open('x') as log:
        subprocess.Popen([sys.executable,str(SOURCE/'tools/watch_exit.py'),str(out)],
                         stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

    def step(variant, work, key, command):
        record_key=variant['name']+'_'+key
        log=out/(record_key+'.log')
        container_path='/workspace/petbot/'+str(work.relative_to(REPOSITORY))
        invocation=[sys.executable,str(work/'tools/run_guarded.py'),'--gpu',str(variant['gpu']),
                    '--reserve-mib',str(plan.get('reserve_mib',8000)),
                    '--workdir',container_path,'--log',str(log),'--',*command]
        with (out/(record_key+'_guard.log')).open('x') as stream:
            child=subprocess.Popen(invocation,stdout=stream,stderr=subprocess.STDOUT)
            with mutex:
                status[record_key]=dict(state='running',pid=child.pid,command=invocation)
                save(out/'eval_status.json',status)
            code=child.wait()
        exit_path=log.with_suffix('.exit.json')
        evidence=json.loads(exit_path.read_text()) if exit_path.exists() else {'missing_exit':True}
        good=code==0 and evidence.get('exit_code')==0
        with mutex:
            status[record_key].update(state='complete_execution_only' if good else 'failed',exit=evidence)
            save(out/'eval_status.json',status)
        return good,evidence

    def run(variant):
        try:
            work=create(variant['name'])
            # Reuse only this engineering gate's source-identified generated asset, never model weights.
            asset_source=REPOSITORY/gate['asset_directory']
            shutil.copytree(asset_source,work/'artifacts')
            with mutex:
                state['variants'][variant['name']]=dict(state='training',gpu=variant['gpu'],seed=variant['seed'],
                                                       source=work.relative_to(REPOSITORY).as_posix(),initialization='random')
                save(out/'launch_status.json',state)
            command=['train.py','--kind','training','--config','configs/round01.json','--run-dir','outputs/train',
                     '--seed',str(variant['seed']),'--iterations',str(variant['iterations']),
                     '--num-envs',str(variant['num_envs']),'--device','cuda:0']
            for attempt in range(3):
                good,evidence=step(variant,work,f'train_attempt{attempt}',command)
                if good: break
                run_status=work/'outputs/train/status.json'
                current=json.loads(run_status.read_text()) if run_status.exists() else {}
                forced=evidence.get('exit_code')==80 or current.get('state') in ('interrupted','failed_oom')
                if not forced or attempt==2: raise RuntimeError(f'Training failed; inspect attempt {attempt}')
                latest=work/'outputs/train/checkpoints/latest.json'
                if latest.exists():
                    checkpoint=json.loads(latest.read_text())['file']
                    command=command[:command.index('--resume-interrupted')] if '--resume-interrupted' in command else command
                    command += ['--resume-interrupted','outputs/train/checkpoints/'+checkpoint]
                elif (work/'outputs/train/run.json').exists():
                    raise RuntimeError('Interrupted before first durable checkpoint; inspect without overwriting run')
                # Finite resource wait for the interrupted same run, not a scheduled agent timer.
                time.sleep(30)
            final=json.loads((work/'outputs/train/status.json').read_text())
            latest=json.loads((work/'outputs/train/checkpoints/latest.json').read_text())
            if final['state']!='complete_execution_only' or final['updates_completed']!=variant['iterations'] or latest['updates_completed']!=variant['iterations']:
                raise RuntimeError('Training exit lacks the required final state/checkpoint')
            checkpoint='outputs/train/checkpoints/'+latest['file']
            with mutex:
                state['variants'][variant['name']]['state']='evaluating'; save(out/'launch_status.json',state)
            results={}
            for scenario in ('rise','static_cycle','low_cycle'):
                if results.get('rise',{}).get('entered_two_wheel',1)==0 and scenario!='rise':
                    with mutex:
                        status[variant['name']+'_eval_'+scenario]=dict(state='skipped',reason='No two-wheel entry; analyze basic learning first')
                        save(out/'eval_status.json',status)
                    continue
                good,evidence=step(variant,work,'eval_'+scenario,['evaluate.py','--run-dir','outputs/train',
                    '--checkpoint',checkpoint,'--output','outputs/eval_'+scenario,'--scenario',scenario,
                    '--count','32','--device','cuda:0'])
                if not good: raise RuntimeError('Evaluation failed: '+scenario)
                results[scenario]=json.loads((work/('outputs/eval_'+scenario)/'result.json').read_text())
            with mutex:
                state['variants'][variant['name']]['state']='execution_finished_acceptance_pending'
                save(out/'launch_status.json',state)
            return dict(name=variant['name'],execution_complete=True,results=results,acceptance_pending=True)
        except BaseException as error:
            with mutex:
                state['variants'].setdefault(variant['name'],{}).update(state='failed',error=repr(error))
                save(out/'launch_status.json',state)
            notify(out,variant['name']+'_failure',thread,
                f"upright新系列{name}的{variant['name']}失败。读取AGENTS.md、docs/current_handoff.md、docs/upright_restart_round01_execution.md及该批状态和实际进程；其他运行可能仍在进行。仅GPU2/3，受迫中断先恢复同一运行；禁止重复启动或加载历史父模型。先核查日志/显存/产物，工程退出0不代表能力通过，无周期定时器或goal。")
            return dict(name=variant['name'],execution_complete=False,error=repr(error),acceptance_pending=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,plan['variants']))
    save(out/'eval_finished.json',dict(results=results,finished_at=time.time(),acceptance_pending=True))
    notify(out,'completion',thread,
        f"upright新系列{name}训练与评估队列已结束。先读AGENTS.md、docs/current_handoff.md、docs/upright_restart_round01_execution.md，核查该批launch_status/eval_status/eval_finished、实际进程、缺失/失败/显存中断与同次恢复证据；分析请求/执行/实测及成功和失败轨迹。仅GPU2/3；新实验从随机初始化，不加载历史权重。先分析再修改继续已授权事件迭代，1000更新demo预算与全部能力要求保持；队列结束不是策略通过，不重复启动，不启用周期定时器或goal。")


if __name__=='__main__': main()
