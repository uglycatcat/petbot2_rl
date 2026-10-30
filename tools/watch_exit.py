"""Wait on one launcher PID; no periodic agent timer or training restart."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
from events import notify, save


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('control',type=Path); args=parser.parse_args()
    out=args.control.resolve()
    lock=(out/'exit_watch.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((out/'launch_status.json').read_text())
    try: fd=os.pidfd_open(state['pid'])
    except ProcessLookupError: fd=None
    if fd is not None:
        try:
            command=(Path('/proc')/str(state['pid'])/'cmdline').read_bytes()
            if b'launch_round01.py' not in command: raise RuntimeError('Launcher identity changed')
            save(out/'exit_watch_pid.json',dict(pid=os.getpid(),launcher=state['pid']))
            select.select([fd],[],[])
        finally: os.close(fd)
    if not (out/'completion_notice.json').exists():
        notify(out,'completion',state['thread'],
               f"upright新系列{state['name']}启动/评估进程已退出，正常完成通知缺失。先读AGENTS.md、docs/current_handoff.md和{out.relative_to(out.parents[2])}状态与实际进程。保留GPU2/3限制；核查缺失/失败/显存中断，继续当前轮，禁止重复启动或加载历史策略。无周期定时器或goal。")
    save(out/'exit_watch_finished.json',dict(finished=True))


if __name__=='__main__': main()
