"""Notify the same session if the batch supervisor exits before successful training."""
import argparse,json,pathlib,subprocess,time,shutil
p=argparse.ArgumentParser();p.add_argument('batch_dir',type=pathlib.Path);a=p.parse_args();out=a.batch_dir
pid=json.loads((out/'maintain_pid.json').read_text())['pid'];proc=pathlib.Path('/proc')/str(pid)/'cmdline'
while proc.exists() and b'maintain_cycle_batch.py' in proc.read_bytes():time.sleep(15)
state=json.loads((out/'launch_status.json').read_text())
if all(j['status']=='complete' for j in state['train'].values()):raise SystemExit(0)
notice=out/'maintenance_failure_notice.json'
if notice.exists():raise SystemExit('Notice already attempted')
notice.write_text(json.dumps({'attempted_at':time.strftime('%FT%T%z'),'supervisor_pid':pid}))
if not state.get('thread') or not shutil.which('codex'):raise SystemExit('No same-session queue available')
message=f"upright_rl_demo批次{state['name']}维护进程已退出但训练未全部完成。读取AGENTS.md、docs/upright_cycle_v1.md及本批maintain_batch.log/launch_status/exit，核查实际进程后修复当前轮，不重复启动或从头重跑；保留固定版本与78%显存守护。"
try:
 result=subprocess.run([shutil.which('codex'),'queue','--thread',state['thread'],'--message',message],capture_output=True,text=True,timeout=120)
 notice.write_text(json.dumps({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}))
except Exception as error:notice.write_text(json.dumps({'submission_unknown':str(error)}))
