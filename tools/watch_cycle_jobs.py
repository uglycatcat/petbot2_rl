"""Read-only live batch supervision with one same-session failure notification."""
import argparse,json,pathlib,subprocess,time,shutil
p=argparse.ArgumentParser();p.add_argument('batch_dir',type=pathlib.Path);a=p.parse_args();out=a.batch_dir
reason=None;state={};deadline=time.monotonic()+900
while True:
 path=out/'launch_status.json'
 if not path.exists():
  if time.monotonic()>deadline:reason='No launch status after15min';break
  time.sleep(5);continue
 state=json.loads(path.read_text());all_complete=len(state['train'])==12
 for name,j in state['train'].items():
  end=out/(name+'.exit.json');proc=pathlib.Path('/proc')/str(j.get('pid',0))/'cmdline'
  if end.exists():
   code=json.loads(end.read_text())['exit_code']
   if code!=0:reason=f'{name} exit{code}';break
  else:
   all_complete=False
   if j.get('pid') and not (proc.exists() and b'run_guarded.py' in proc.read_bytes()):reason=f'{name} guard disappeared without exit record';break
 if reason:break
 if all_complete and state.get('eval_pid'):raise SystemExit(0)
 if not state.get('eval_pid'):
  pid=json.loads((out/'launcher_pid.json').read_text())['pid'];proc=pathlib.Path('/proc')/str(pid)/'cmdline'
  if not (proc.exists() and b'launch_cycle.py' in proc.read_bytes()):reason='Launcher exited before evaluation queue attached';break
 time.sleep(10)
notice=out/'live_failure_notice.json'
if notice.exists():raise SystemExit('Notice already attempted')
notice.write_text(json.dumps({'reason':reason,'attempted_at':time.strftime('%FT%T%z')}))
if state.get('thread') and shutil.which('codex'):
 try:
  r=subprocess.run([shutil.which('codex'),'queue','--thread',state['thread'],'--message',f"upright_rl_demo批次{state['name']}监督发现中断：{reason}。读取AGENTS.md及本批launch_status/exit/launcher.log，核查实际进程后恢复当前轮；不重复启动、不覆盖固定版本。"],capture_output=True,text=True,timeout=120)
  notice.write_text(json.dumps({'reason':reason,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
 except Exception as error:notice.write_text(json.dumps({'reason':reason,'submission_unknown':str(error)}))
