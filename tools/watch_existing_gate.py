"""Attach a finite exit event to already running guarded engineering jobs.

Never launches or retries simulation. pidfds identify existing processes, and
exclusive state files prevent duplicate attachment. Final acceptance is left
to the resumed agent; process success is not a declaration of task success.
"""
import argparse
import fcntl
import json
import os
import pathlib
import select
import shutil
import subprocess
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def attach(out, names, queue_name, context_doc='docs/upright_handoff_alignment_v18.md'):
    lock = (out/'gate_watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out/'launch_status.json').exists(), 'Existing attachment; inspect before recovery'
    thread = os.environ.get('CODEX_THREAD_ID')
    executable = shutil.which('codex')
    assert thread and executable, 'Same-session queue unavailable'
    context = pathlib.PurePosixPath(context_doc)
    assert context.parts[0]=='docs' and '..' not in context.parts and context.suffix=='.md'
    jobs = json.loads((out/'jobs.json').read_text())
    assert names and len(names) == len(set(names)) and all(n in jobs for n in names)
    descriptors = {}
    state = {}
    # Open/verify all identities before publishing an attachment.
    for name in names:
        pid = jobs[name]['pid']
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            fd = None
        if fd is not None:
            try:
                cmd = (pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
            except FileNotFoundError:
                cmd = []
            expected_log = str(out/(name+'.log')).encode()
            if cmd and (not any(x.endswith(b'/run_guarded.py') for x in cmd) or expected_log not in cmd):
                os.close(fd)
                raise RuntimeError('Guard PID identity mismatch: '+name)
            descriptors[fd] = name
        state[name] = dict(status='running' if fd is not None else 'exited', pid=pid, gpu=jobs[name]['gpu'])
    save(out/'launch_status.json', dict(name=queue_name, eval_pid=os.getpid(),
         evaluator_entrypoint='watch_existing_gate.py', thread=thread, attached_existing_only=True,
         expected_jobs=len(names), jobs=names, context_doc=context_doc, started_at=time.strftime('%FT%T%z')))
    save(out/'eval_status.json', state)
    with (out/'eval_watch.log').open('x') as log:
        subprocess.Popen(['python3', str(pathlib.Path(__file__).with_name('watch_cycle_evaluation.py')), str(out)],
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def record(name):
        item = state[name]
        try:
            end = json.loads((out/(name+'.exit.json')).read_text())
            ok = end['exit_code'] == 0 and not end['guard_stopped'] and not end.get('physics_error')
            item.update(status='complete' if ok else 'failed', exit=end)
        except Exception as error:
            item.update(status='missing_exit', error=repr(error))
        item['finished_at'] = time.strftime('%FT%T%z')
        save(out/'eval_status.json', state)

    for name in names:
        if state[name]['status'] == 'exited':
            record(name)
    while descriptors:
        ready, _, _ = select.select(list(descriptors), [], [])
        for fd in ready:
            name = descriptors.pop(fd); os.close(fd); record(name)
    complete = sum(x['status'] == 'complete' for x in state.values())
    save(out/'eval_finished.json', dict(expected=len(names), complete=complete,
         failed=len(names)-complete, acceptance_pending=True, finished_at=time.strftime('%FT%T%z')))
    message = (
        f'upright_rl_demo工程验收{queue_name}进程已结束。先读AGENTS.md、docs/current_handoff.md、'
        f'{context_doc}，核查runs/{out.name}的launch_status/eval_status/'
        'eval_finished及实际进程、短训/重载结果和物理约束。检查是否发生显存中断及恢复，'
        '不能遗漏失败、假设退出0代表通过或重复启动。这是工程验收，不是正式PPO批次终局。'
        '先完成验收与结果分析，再决定针对起立缺陷的训练或MuJoCo闭环；'
        '网络导出不是sim2sim通过。继续今晚已授权事件迭代，'
        '稳定优先，不恢复周期定时器或goal。')
    save(out/'completion_notice.json', dict(submission_unknown='attempt_started'))
    try:
        result = subprocess.run([executable, 'queue', '--thread', thread, '--message', message],
                                capture_output=True, text=True, timeout=120)
        notice = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except Exception as error:
        notice = dict(submission_unknown=repr(error))
    save(out/'completion_notice.json', notice)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--context-doc', default='docs/upright_handoff_alignment_v18.md')
    parser.add_argument('directory', type=pathlib.Path)
    parser.add_argument('jobs', nargs='+')
    args = parser.parse_args()
    attach(args.directory.resolve(), args.jobs, args.name, args.context_doc)
