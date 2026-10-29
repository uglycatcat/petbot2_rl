"""One-batch process-exit hook: notify this session if its evaluator disappears.

Linux pidfd waits for the actual evaluator exit, with no recurring timer and no
training restart. Normal completion is already notified by evaluate_cycle.py.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_dir', type=pathlib.Path)
    args = parser.parse_args()
    out = args.batch_dir.resolve()
    lock = (out / 'eval_watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result_path = out / 'eval_watch_result.json'
    if result_path.exists():
        raise RuntimeError('Exit hook already handled this evaluator; inspect before reattaching')
    state = json.loads((out / 'launch_status.json').read_text())
    pid = state['eval_pid']
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        fd = None
    if fd is not None:
        cmdline_path = pathlib.Path('/proc') / str(pid) / 'cmdline'
        try:
            command = cmdline_path.read_bytes()
        except FileNotFoundError:
            command = b''
        entrypoint = state.get('evaluator_entrypoint', 'evaluate_cycle.py')
        if entrypoint not in ('evaluate_cycle.py', 'evaluate_candidates.py', 'watch_existing_gate.py', 'launch_scratch.py'):
            os.close(fd)
            raise RuntimeError('Unknown evaluator entrypoint')
        if command and (entrypoint.encode() not in command or state['name'].encode() not in command):
            os.close(fd)
            raise RuntimeError('Evaluator PID identity mismatch')
        (out / 'eval_watch_pid.json').write_text(json.dumps(dict(pid=os.getpid(), evaluator_pid=pid)))
        try:
            select.select([fd], [], [])
        finally:
            os.close(fd)
    notice_path = out / 'completion_notice.json'
    if notice_path.exists():
        notice = json.loads(notice_path.read_text())
        if notice.get('returncode') == 0 or 'submission_unknown' in notice:
            # Unknown delivery needs manual reconciliation, never blind duplicate delivery.
            result_path.write_text(json.dumps(dict(evaluator_pid=pid, existing_notice=notice,
                action='no_duplicate_notification', finished_at=time.strftime('%FT%T%z')), indent=2))
            return
    report = dict(evaluator_pid=pid, finished_at=time.strftime('%FT%T%z'))
    result_path.write_text(json.dumps({**report, 'action': 'notification_attempt_started'}, indent=2))
    executable = shutil.which('codex')
    if not executable or not state.get('thread'):
        raise RuntimeError('Same-session notification unavailable')
    message = (
        f"upright_rl_demo批次{state['name']}评估进程已退出，正常完成通知缺失或失败。"
        "读取AGENTS.md、docs/current_handoff.md与本批launch_status/eval_status/eval_finished、"
        "实际进程及日志。先判断正常结束、评估失败或中断；未完成则恢复当前轮，禁止重复启动。"
        "若结果齐全，按最新稳定优先与基本遥控需求分析后继续迭代；旧瞬时精度不作为主要否决条件。保持固定版本、78%显存守护与周期定时器停用。")
    try:
        result = subprocess.run([executable, 'queue', '--thread', state['thread'], '--message', message],
                                capture_output=True, text=True, timeout=120)
        report.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except Exception as error:
        report['submission_unknown'] = str(error)
    result_path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
