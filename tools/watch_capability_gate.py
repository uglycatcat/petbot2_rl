"""Observe a finite pre-launched engineering queue; never start/retry a job."""
import argparse,fcntl,json,os,pathlib,shutil,subprocess,time
p=argparse.ArgumentParser();p.add_argument('directory',type=pathlib.Path);a=p.parse_args();out=a.directory
lock=(out/'gate_watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=json.loads((out/'launch_status.json').read_text());jobs=state['train'];results={}
assert not (out/'eval_finished.json').exists()
while len(results)<len(jobs):
 for name,j in jobs.items():
  if name in results:continue
  proc=pathlib.Path('/proc')/str(j['pid'])/'cmdline';end=out/(name+'.exit.json')
  cmd=proc.read_bytes() if proc.exists() else b''
  alive=b'run_guarded.py' in cmd and str(out/(name+'.log')).encode() in cmd
  if alive:continue
  record=json.loads(end.read_text()) if end.exists() else {}
  ok=record.get('exit_code')==0 and not any(record.get(k) for k in ('guard_stopped','physics_error','runtime_error'))
  results[name]=dict(status='complete' if ok else 'failed',exit=record,acceptance_pending=True)
 (out/'eval_status.json').write_text(json.dumps(results,indent=2));time.sleep(3)
(out/'eval_finished.json').write_text(json.dumps(dict(expected=len(jobs),complete=sum(x['status']=='complete' for x in results.values()),failed=sum(x['status']!='complete' for x in results.values()),acceptance_pending=True,finished_at=time.strftime('%FT%T%z')),indent=2))
if state.get('thread') and shutil.which('codex'):
 message=f"upright_rl_demo工程验收{state['name']}队列已结束。读取AGENTS.md、docs/current_handoff.md、docs/upright_cycle_v25.md及runs/{out.name}状态，核查实际进程、日志、显存中断、失败、短训/重载与物理轨迹；不得把退出0或队列结束当通过，不重复启动。保留最新1000iteration正式训练授权和落地仅回默认髋位、起立少调整约束。先验收后继续当前轮事件迭代，不恢复周期定时器或goal。"
 try:
  r=subprocess.run([shutil.which('codex'),'queue','--thread',state['thread'],'--message',message],capture_output=True,text=True,timeout=60)
  notice=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 except Exception as e:notice=dict(submission_unknown=str(e))
 (out/'completion_notice.json').write_text(json.dumps(notice,indent=2))
