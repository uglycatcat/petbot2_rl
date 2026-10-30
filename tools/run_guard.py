"""Host supervisor: process exit + artifact + physical-error audit, with GPU protection."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

FATAL=re.compile(r'Traceback \(most recent call last\)|CUDA out of memory|illegal memory access|\[Error\].*(?:physx|physics)',re.I)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--container',default='petbot-step-train')
    p.add_argument('--workdir',default='/workspace/demo_project/step_rl_demo')
    p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--memory-fraction',type=float,default=.78)
    p.add_argument('--timeout',type=float,default=1800.)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--status',type=Path,required=True,help='Host path to expected JSON artifact')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command: p.error('Provide Python module/script arguments after --')
    if a.log.exists(): p.error('Use a fresh log path')
    if not 0<a.memory_fraction<1: p.error('memory-fraction must be between 0 and 1')
    gpu_out=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}',
        '--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    used,total=map(float,gpu_out.strip().split(','))
    if used/total>=a.memory_fraction: p.error('Selected GPU already exceeds memory threshold')
    token=uuid.uuid4().hex;pidfile=f'/tmp/step-{token}.pid'
    bootstrap=('import os,sys;from pathlib import Path;os.setpgid(0,0);'
               'Path(sys.argv[1]).write_text(str(os.getpid()));'
               'os.environ["STEP_RUN_TOKEN"]=sys.argv[2];'
               'os.execv("/isaac-sim/python.sh",["/isaac-sim/python.sh"]+sys.argv[3:])')
    kill=('import os,signal,sys;from pathlib import Path;p=Path(sys.argv[1]);'
          'pid=int(p.read_text()) if p.exists() else -1;'
          'env=Path(f"/proc/{pid}/environ");'
          'valid=pid>1 and env.exists() and ("STEP_RUN_TOKEN="+sys.argv[2]).encode() in env.read_bytes().split(b"\\0");'
          'os.killpg(pid,signal.SIGTERM) if valid else None')
    a.log.parent.mkdir(parents=True,exist_ok=True)
    reason=None;started=time.monotonic();peak=used
    with a.log.open('w') as log:
        child=subprocess.Popen(['docker','exec','-w',a.workdir,a.container,'python3','-c',bootstrap,pidfile,token,*command],stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if time.monotonic()-started>a.timeout: reason='timeout';break
                gpu_out=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}',
                    '--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
                used,total=map(float,gpu_out.strip().split(','));peak=max(peak,used)
                if used/total>=a.memory_fraction: reason='gpu_memory_threshold';break
                time.sleep(2.)
        except BaseException:
            reason='supervisor_interrupted'
            raise
        finally:
            if reason:
                subprocess.run(['docker','exec',a.container,'python3','-c',kill,pidfile,token],check=False)
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    subprocess.run(['docker','exec',a.container,'python3','-c',kill.replace('SIGTERM','SIGKILL'),pidfile,token],check=False)
                    child.wait(timeout=15)
    matches=FATAL.findall(a.log.read_text(errors='replace'))
    artifact=json.loads(a.status.read_text()) if a.status.exists() else {}
    valid=(reason is None and child.returncode==0 and not matches and artifact.get('status') in ('complete','interrupted','passed'))
    result=dict(valid=valid,exit_code=child.returncode,reason=reason,physical_or_python_errors=matches,
                artifact_status=artifact.get('status'),gpu_peak_mib=peak,elapsed_s=time.monotonic()-started)
    a.log.with_suffix(a.log.suffix+'.guard.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    return 0 if valid else 82

if __name__=='__main__': raise SystemExit(main())
