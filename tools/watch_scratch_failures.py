"""Finite batch failure hook; no training restarts or periodic agent tasks."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

out=Path(sys.argv[1]).resolve()
lock=(out/'failure_watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
launch=json.loads((out/'launch_status.json').read_text())
pid=launch['eval_pid']
try:parent=os.pidfd_open(pid)
except ProcessLookupError:parent=None
if parent is not None:
    cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
    assert b'launch_scratch.py' in cmd and launch['name'].encode() in cmd
(out/'failure_watch_pid.json').write_text(json.dumps(dict(pid=os.getpid(),batch_pid=pid)))
executable=shutil.which('codex');assert executable and launch['thread']
while True:
    status=json.loads((out/'eval_status.json').read_text())
    for name,item in status.items():
        notice=out/(name+'_failure_notice.json')
        if item['status']!='failed' or notice.exists():continue
        notice.write_text(json.dumps(dict(submission_unknown='attempt_started')))
        message=(f"upright_rl_demo从零批次{launch['name']}的{name}失败。"
            "先读AGENTS.md、docs/current_handoff.md、docs/upright_cycle_v26.md并核查该批实际进程和状态。"
            "其他运行可能仍在进行，不能重复启动或覆盖固定副本。检查显存/日志/检查点；受迫中断先恢复同一运行，"
            "仅修正工具问题不能变成加载旧父策略的新实验。保留1000更新及所有最新需求；继续事件驱动纠正，禁止周期定时器和goal。")
        try:
            r=subprocess.run([executable,'queue','--thread',launch['thread'],'--message',message],capture_output=True,text=True,timeout=120)
            result=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except Exception as error:result=dict(submission_unknown=repr(error))
        notice.write_text(json.dumps(result,indent=2)+'\n')
    if (out/'eval_finished.json').exists():break
    if parent is None:break
    # This is a finite process monitor, like the memory guard, not a scheduled
    # wakeup of the agent. Normal completion has a separate one-shot notice.
    import select
    if select.select([parent],[],[],1.)[0]:break
if parent is not None:os.close(parent)
(out/'failure_watch_finished.json').write_text(json.dumps(dict(finished_at=time.strftime('%FT%T%z'))))
