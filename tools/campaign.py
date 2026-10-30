"""Host-only step batch lifecycle and completion-event delivery to the same Codex session."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from step_demo.runtime import load_training
from step_demo.review import summarize


def read(path):return json.loads(Path(path).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def safe_relative(base,value):
    path=(Path(base)/value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(Path(base).resolve()):
        raise ValueError('Paths must remain inside this fixed round directory')
    return path


def validate_plan(plan,folder):
    if plan.get('schema')!='step_batch_v1' or plan.get('kind') not in ('engineering','formal'):
        raise ValueError('Unknown plan schema/kind')
    if not plan.get('hypothesis') or not plan.get('jobs'): raise ValueError('A plan needs a hypothesis and jobs')
    ids=set();gpus=set()
    for job in plan['jobs']:
        if job['gpu'] not in (0,1):raise ValueError('Step may use only physical GPU 0/1')
        if job['gpu'] in gpus or job['id'] in ids:raise ValueError('One job per GPU and unique job IDs required')
        gpus.add(job['gpu']);ids.add(job['id']);safe_relative(folder,job['id'])
        config=safe_relative(Path(folder)/'source',job['training_config'])
        load_training(config)
        if type(job['eval_envs']) is not int or not 1<=job['eval_envs']<=512:raise ValueError('Invalid evaluation size')
    if plan.get('round_number',0)>1:
        if not plan.get('previous_review') or not plan.get('decision'):
            raise ValueError('A later round requires evidence and an explicit optimization decision')
        if not plan['decision'].get('change') or not plan['decision'].get('reason'):
            raise ValueError('Do not blindly repeat the preceding configuration')


def queue_event(control_path,event_path,message):
    """Write before delivery. Uncertain deliveries are preserved and never blindly retried."""
    control=read(control_path);event_path=Path(event_path)
    if event_path.exists():return read(event_path)
    if not control.get('enabled'):return dict(status='disabled')
    data=dict(status='submitting',created_at=stamp(),message=message)
    write(event_path,data)
    try:
        result=subprocess.run([control['codex'],'queue','--thread',control['thread'],'--message',message],
            cwd=control['cwd'],capture_output=True,text=True,timeout=60)
        data.update(status='queued' if result.returncode==0 else 'failed',exit_code=result.returncode,
                    stdout=result.stdout,stderr=result.stderr,finished_at=stamp())
    except subprocess.TimeoutExpired:
        data.update(status='delivery_unknown',finished_at=stamp())
    except Exception:
        data.update(status='failed',error=traceback.format_exc(),finished_at=stamp())
    write(event_path,data);return data


def enabled(control):return bool(read(control).get('enabled'))


def gpu_ready(gpu,minimum_free_mib=3400):
    result=subprocess.check_output(['nvidia-smi',f'--id={gpu}',
        '--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    used,total=map(float,result.strip().split(','))
    return .78*total-used>=minimum_free_mib


def container_directory(folder,control):
    host=Path(read(control)['runs_root']).resolve()
    relative=Path(folder).resolve().relative_to(host)
    return str(Path('/workspace/runs')/relative/'source')


def job_run(job,folder,control):
    folder=Path(folder);job_dir=safe_relative(folder,job['id']);job_dir.mkdir(exist_ok=True)
    state_path=job_dir/'job_status.json'
    if state_path.exists() and read(state_path).get('status')=='complete':return read(state_path)
    state=dict(job=job['id'],gpu=job['gpu'],status='waiting_for_gpu',updated_at=stamp())
    write(state_path,state)
    while enabled(control) and not gpu_ready(job['gpu']):time.sleep(15.)
    if not enabled(control):state.update(status='cancelled_by_user');write(state_path,state);return state
    source=folder/'source';guard=source/'tools/run_guard.py';train=job_dir/'train'
    working=container_directory(folder,control)
    common=['--asset-dir','../asset','--model-dir','../../../robot_description']
    def guarded(phase,command,status):
        attempt=1
        while (job_dir/f'{phase}_{attempt:02}.log').exists():attempt+=1
        log=job_dir/f'{phase}_{attempt:02}.log'
        state.update(status=phase,updated_at=stamp(),log=str(log));write(state_path,state)
        args=[sys.executable,str(guard),'--gpu',str(job['gpu']),'--workdir',working,
              '--timeout','86400','--control',str(control),'--log',str(log),'--status',str(status),'--',*command]
        result=subprocess.run(args,cwd=source)
        guard_result=read(log.with_suffix('.log.guard.json')) if log.with_suffix('.log.guard.json').exists() else dict(valid=False,reason='guard_start_failed')
        if result.returncode or not guard_result.get('valid'):
            raise RuntimeError(json.dumps(dict(phase=phase,guard=guard_result),ensure_ascii=False))
        return guard_result
    try:
        train_status=read(train/'status.json') if (train/'status.json').exists() else {}
        if train_status.get('status')!='complete':
            command=['-m','step_demo','train','--run-dir',f'../{job["id"]}/train',
                     '--training-config',job['training_config'],*common]
            if train.exists():
                if not (train/'latest.json').exists():raise RuntimeError('Previous attempt lacks a checkpoint; preserve it for agent diagnosis')
                command.append('--resume')
            guarded('training',command,train/'status.json')
        evaluation=job_dir/'evaluation'
        if not (evaluation/'status.json').exists() or read(evaluation/'status.json').get('status')!='complete':
            if evaluation.exists():
                count=1
                while (job_dir/f'evaluation_failed_{count:02}').exists():count+=1
                evaluation.rename(job_dir/f'evaluation_failed_{count:02}')
            command=['-m','step_demo','evaluate','--run-dir',f'../{job["id"]}/train',
                '--output',f'../{job["id"]}/evaluation','--eval-envs',str(job['eval_envs']),*common]
            guarded('evaluating',command,evaluation/'status.json')
        state.update(status='analyzing',updated_at=stamp());write(state_path,state)
        report=summarize(train,evaluation);write(job_dir/'analysis.json',report)
        state.update(status='complete',updated_at=stamp(),result=report['evaluation']['outcomes'],
                     analysis=str(job_dir/'analysis.json'))
    except Exception:
        state.update(status='failed' if enabled(control) else 'cancelled_by_user',updated_at=stamp(),error=traceback.format_exc())
    write(state_path,state);return state


def run(folder,control):
    folder=Path(folder).resolve();control=Path(control).resolve();cfg=read(control)
    if cfg.get('project')!='step' or cfg.get('allowed_gpus')!=[0,1]:raise ValueError('Invalid campaign scope')
    if not enabled(control):raise ValueError('Campaign disabled')
    plan=read(folder/'plan.json');validate_plan(plan,folder)
    lock=(control.parent/'active.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(control.parent/'active.json',dict(round=str(folder),pid=os.getpid(),started_at=stamp(),status='running'))
    state=dict(status='running',started_at=stamp(),plan_sha256=hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest(),jobs=[])
    write(folder/'batch_status.json',state)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(job_run,job,folder,control) for job in plan['jobs']]
            for future in as_completed(futures):
                result=future.result()
                state['jobs'].append(result);write(folder/'batch_status.json',state)
                if len(state['jobs'])<len(plan['jobs']) and enabled(control):
                    queue_event(control,folder/f'job_{result["job"]}_event.json',
                        f'[step单路结果事件] {folder}/{result["job"]}/job_status.json已有结果。'
                        f'先读取{control.parent}/active.json、{folder}/agent_review.json（若有）、{folder}/batch_status.json及{control}。'
                        '若是已处理旧事件且更新批次已运行，只补记接收，不重复发批或中止新训练。'
                        '用户授权持续迭代，仅GPU0/1。立即审查本路真实轨迹/analysis.json和失败原因，准备改进；'
                        '同批其它路仍可运行，不能修改固定副本或重复发批。等整批完成事件后汇总并验证、推送下一轮。')
        state.update(status='complete' if all(job['status']=='complete' for job in state['jobs']) else 'failed',finished_at=stamp())
        if not enabled(control):state['status']='cancelled_by_user'
        write(folder/'batch_status.json',state)
        if enabled(control):
            event_kind='工程检查完成' if plan['kind']=='engineering' else '正式批次完成/异常'
            message=(f'[step自动迭代事件：{event_kind}] 固定目录 {folder}；控制文件 {control}。'
                f'先核对{control.parent}/active.json、enabled、agent_review.json（若有）、batch_status.json、每路analysis.json、日志和实际进程。'
                '若事件属于已处理的旧轮次且更新批次已运行，只补记接收，不重复发批或中止新训练。'
                '只负责step，物理GPU仅0/1。用户已授权持续分析、修改优化、工程验证并重新从零训练。'
                '检查真实阶段、请求/执行/实测关节、接触、运动和成功/失败轨迹，记录结论与下一轮修改假设；'
                '任何无效物理或失败工程不得作策略结论。新配置/源码先过容器门槛、提交推送、冻结，再启动下一批并延续同一事件链。'
                '不得加载历史权重、修改源机器人、使用其它GPU、启动sim2sim或其它项目。'
                '若本事件为工程连通性测试且正式批次已经启动，只核对并记录接收，不重复发批。')
            queue_event(control,folder/'continuation_event.json',message)
    except Exception:
        state.update(status='controller_failed',error=traceback.format_exc(),finished_at=stamp())
        write(folder/'batch_status.json',state)
        if enabled(control):queue_event(control,folder/'controller_failure_event.json',
            f'[step控制器异常] 检查{folder}/batch_status.json和{control}，仅step/GPU0/1；修复后继续用户授权迭代，避免重复运行。')
        raise
    finally:
        write(control.parent/'active.json',dict(round=str(folder),pid=os.getpid(),finished_at=stamp(),status=state['status']))
        lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('init','run','stop','canary'))
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--round-dir',type=Path)
    a=p.parse_args();control=a.control.resolve()
    if a.command=='init':
        if control.exists():raise ValueError('Control already exists; do not overwrite authorization or active state')
        thread=os.environ.get('CODEX_THREAD_ID')
        if not thread or not shutil.which('codex'):raise RuntimeError('Current session identity and codex queue are required')
        control.parent.mkdir(parents=True,exist_ok=True)
        root=PROJECT.parents[1]
        write(control,dict(project='step',enabled=True,allowed_gpus=[0,1],thread=thread,codex=shutil.which('codex'),
            cwd=str(root),runs_root=str(root/'demo_project/runs'),created_at=stamp(),authorization='2026-10-30 user: train/analyze/optimize/repeat; only GPU0/1'))
    elif a.command=='stop':
        cfg=read(control);cfg.update(enabled=False,stopped_at=stamp());write(control,cfg)
    elif a.command=='canary':
        result=queue_event(control,control.parent/'canary_event.json',
            f'[step事件连通性检查] 用户已授权自动迭代。仅核对{control.parent}/canary_event.json，写同目录canary_ack.json确认本会话收到；继续当前工作，不重复发批、不改任务范围。')
        print(json.dumps(result,ensure_ascii=False))
    else:
        if not a.round_dir:p.error('--round-dir required')
        run(a.round_dir,control)

if __name__=='__main__':main()
