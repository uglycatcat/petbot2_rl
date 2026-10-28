"""Finite, explicit evaluation-only plan with fixed code and same-session completion.

No training, parameter selection, periodic scheduling or automatic retries.
Interrupted plans retain their state and require inspection before continuation.
"""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time

from cycle_evaluation_contract import validate
from physics_log import PhysicsLogReader


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def command(root, out, plan, job, gpu):
    fixed = root / plan['fixed']
    checkpoint = pathlib.Path(job['checkpoint'])
    relative = checkpoint.relative_to('demo_project/runs')
    extra = job.get('args', [])
    forbidden = ('--checkpoint', '--initialize', '--log_dir', '--iterations',
                 '--evaluate', '--probe', '--device', '--num_envs', '--allow-dirty')
    assert not any(x.startswith(forbidden) for x in plan['common_args']+extra)
    assert job['mode'] in ('checkpoint', 'untuned_parent')
    entry = job.get('entrypoint', 'train_cycle.py')
    assert entry in ('train_cycle.py', 'tools/probe_landing_state.py', 'tools/probe_rise_authority.py', 'tools/probe_balance_refresh.py', 'tools/probe_transition_speed.py')
    flags = (['--evaluate', '--checkpoint'] if job['mode'] == 'checkpoint'
             else ['--probe', '--initialize-cycle-from'])
    return ['python3', str(fixed/'tools/run_guarded.py'), '--container', plan['container'],
            '--gpu', str(gpu), '--reserve-mib', '4000', '--entrypoint', entry,
            '--workdir', '/workspace/runs/'+str(fixed.relative_to(root/'demo_project/runs')),
            '--log', str(out/(job['name']+'.log')), '--headless', '--device', 'cuda:0',
            '--num_envs', '64', '--seed', str(job['seed']), *flags,
            '/workspace/runs/'+str(relative), '--log_dir',
            '/workspace/runs/'+str((out/job['name']).relative_to(root/'demo_project/runs')),
            *plan['common_args'], *extra]


def verify_plan(root, plan):
    fixed = root / plan['fixed']
    context = pathlib.PurePosixPath(plan.get('context_doc', 'docs/upright_cycle_batch17_results.md'))
    assert context.parts[0] == 'docs' and '..' not in context.parts and context.suffix == '.md'
    assert len({j['name'] for j in plan['jobs']}) == len(plan['jobs']) > 0
    assert 1<=plan.get('parallelism',4)<=4
    assert set(plan.get('excluded_gpus',[]))<=set(range(4))
    assert all(pathlib.Path(j['name']).name == j['name'] and j['name'] not in ('.', '..') for j in plan['jobs'])
    assert subprocess.check_output(['git', '-C', str(fixed), 'rev-parse', 'HEAD'], text=True).strip() == plan['commit']
    assert not subprocess.check_output(['git', '-C', str(fixed), 'status', '--porcelain'], text=True).strip()
    # Simulator/task source remains accepted; named diagnostic wrappers declare
    # any policy intervention explicitly in their evaluation contract.
    assert not subprocess.check_output(['git', '-C', str(fixed), 'diff', plan['accepted_task_commit'],
                                       plan['commit'], '--', 'train_cycle.py', 'upright'], text=True)
    for job in plan['jobs']:
        assert hashlib.sha256((root/job['checkpoint']).read_bytes()).hexdigest() == job['checkpoint_sha256']
        command(root, root/'demo_project/runs'/('_'+plan['name']), plan, job, 2)


def run(root, plan):
    out = root/'demo_project/runs'/('_'+plan['name'])
    out.mkdir(exist_ok=True)
    lock = (out/'eval.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'launch_status.json').exists() or (out/'eval_status.json').exists():
        raise RuntimeError('Existing evaluation plan: inspect before recovery; never duplicate')
    verify_plan(root, plan)
    thread = os.environ.get('CODEX_THREAD_ID')
    if not thread or not shutil.which('codex'):
        raise RuntimeError('Same-session completion hook unavailable')
    save(out/'launch_status.json', dict(name=plan['name'], commit=plan['commit'],
         eval_pid=os.getpid(), evaluator_entrypoint='evaluate_candidates.py',
         evaluation_only=True, expected_evaluations=len(plan['jobs']), thread=thread,
         started_at=time.strftime('%FT%T%z'), plan=plan))
    with (out/'eval_watch.log').open('x') as log:
        subprocess.Popen(['python3', str(root/plan['fixed']/'tools/watch_cycle_evaluation.py'), str(out)],
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state = {j['name']:dict(status='pending') for j in plan['jobs']}
    save(out/'eval_status.json', state)
    running = {}
    deadline = time.monotonic()+12*3600
    while any(x['status'] == 'pending' for x in state.values()) or running:
        for name, active in list(running.items()):
            code = active['process'].poll()
            if code is None:
                continue
            item = state[name]
            item.update(status='failed', exit_code=code, finished_at=time.strftime('%FT%T%z'))
            try:
                end = json.loads((out/(name+'.exit.json')).read_text())
                assert code == end['exit_code'] == 0 and not end['guard_stopped'] and not end.get('physics_error')
                assert PhysicsLogReader(out/(name+'.log')).check() is None
                result = out/name/'evaluation.json'
                assert result.stat().st_mtime >= active['started']
                data = validate(json.loads(result.read_text()))
                assert data['count'] == 64 and data['unfinished'] == 0
                item.update(status='complete', result=data)
            except Exception as error:
                item['validation_error'] = repr(error)
            del running[name]
            save(out/'eval_status.json', state)
        if time.monotonic() >= deadline:
            for item in state.values():
                if item['status'] == 'pending':
                    item['status'] = 'resource_timeout'
            save(out/'eval_status.json', state)
        for job in plan['jobs']:
            if state[job['name']]['status'] != 'pending' or len(running) >= plan.get('parallelism',4):
                continue
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used',
                                           '--format=csv,noheader,nounits'], text=True)
            memory = {int(i):(int(t),int(u)) for i,t,u in (line.split(',') for line in raw.splitlines())}
            occupied = {x['gpu'] for x in running.values()}
            gpu = next((i for i in (2,3,1,0) if i not in occupied and i not in plan.get('excluded_gpus',[]) and memory[i][1]+4000 < .78*memory[i][0]), None)
            if gpu is None:
                break
            name = job['name']; cmd = command(root, out, plan, job, gpu); started = time.time()
            with (out/(name+'_guard.log')).open('x') as log:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            running[name] = dict(process=process, gpu=gpu, started=started)
            state[name] = dict(status='running', gpu=gpu, pid=process.pid, command=cmd,
                               started_at=time.strftime('%FT%T%z'))
            save(out/'eval_status.json', state)
        if running or any(x['status'] == 'pending' for x in state.values()):
            time.sleep(5)
    complete = sum(x['status'] == 'complete' for x in state.values())
    save(out/'eval_finished.json', dict(expected=len(state), complete=complete,
         failed=len(state)-complete, finished_at=time.strftime('%FT%T%z')))
    context = pathlib.PurePosixPath(plan.get('context_doc', 'docs/upright_cycle_batch17_results.md'))
    assert context.parts[0] == 'docs' and '..' not in context.parts and context.suffix == '.md'
    message = (f"upright_rl_demo固定策略验证{plan['name']}队列已结束。先读AGENTS.md、docs/current_handoff.md、"
               f"{context}及本验证计划，核查launch_status/eval_status/eval_finished、实际进程与缺失/失败。"
               "按稳定优先分析阶段失败与基本遥控；区分独立验证和开发干预，不能以队列结束认定任务通过。"
               "这是固定策略队列，不是新PPO批次。继续已授权事件迭代，先分析再决定修改/训练或MuJoCo验证；"
               "禁止重复启动，不启用周期定时器或goal。")
    try:
        result = subprocess.run([shutil.which('codex'), 'queue', '--thread', thread, '--message', message],
                                capture_output=True, text=True, timeout=120)
        notice = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except Exception as error:
        notice = dict(submission_unknown=str(error))
    save(out/'completion_notice.json', notice)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--plan', type=pathlib.Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan['name'] == args.name
    run(args.root.resolve(), plan)
