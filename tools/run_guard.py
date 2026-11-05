"""Host supervisor: process exit + artifact + physical-error audit, with GPU protection."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.gpu_resources import memory_by_container, pressure_victim

FATAL=re.compile(r'Traceback \(most recent call last\)|CUDA out of memory|illegal memory access|\[Error\].*(?:physx|physics)',re.I)

def control_enabled(path):
    cfg = json.loads(Path(path).read_text())
    return bool(cfg.get('enabled')) and not cfg.get('paused_by_user', False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--container')
    p.add_argument('--memory-peer',action='append',default=[],help='Same-card STEP container; second slot yields first under memory pressure')
    p.add_argument('--workdir',default='/workspace/demo_project/step_rl_demo')
    p.add_argument('--gpu',type=int,choices=range(4),default=1)
    p.add_argument('--memory-fraction',type=float,default=.78)
    p.add_argument('--reserve-mib',type=int,default=2048)
    p.add_argument('--timeout',type=float,default=1800.,help='Seconds; 0 lets the training update budget determine completion')
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--status',type=Path,required=True,help='Host path to expected JSON artifact')
    p.add_argument('--control',type=Path,help='Campaign JSON; enabled=false stops only this run process group')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    a.container=a.container or f'petbot-step-gpu{a.gpu}'
    if not command: p.error('Provide Python module/script arguments after --')
    if a.log.exists(): p.error('Use a fresh log path')
    if not 0<a.memory_fraction<1: p.error('memory-fraction must be between 0 and 1')
    if a.reserve_mib < 1024: p.error('Keep at least 1024 MiB free for shared-server headroom')
    if a.timeout < 0:p.error('timeout must be nonnegative')
    expected_uuid=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    for name in [a.container,*a.memory_peer]:
        container=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
        if not container['State']['Running']:p.error('Dedicated step container is not running')
        requests=container['HostConfig'].get('DeviceRequests') or []
        devices=[item for request in requests for item in (request.get('DeviceIDs') or [])]
        if len(devices)!=1 or devices[0] not in (str(a.gpu),expected_uuid):
            p.error('Container GPU exposure must exactly match the selected physical GPU')
        if container['Config'].get('Labels',{}).get('petbot.project')!='step': p.error('Use a dedicated step container')
    if a.control and not control_enabled(a.control): p.error('Campaign is disabled or paused')
    if a.control and a.gpu not in json.loads(a.control.read_text()).get('allowed_gpus',[]):
        p.error('Selected GPU is outside this control file authorization')
    gpu_out=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}',
        '--query-gpu=memory.used,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
    used,total,free=map(float,gpu_out.strip().split(','))
    if used/total>=a.memory_fraction or free<a.reserve_mib:
        p.error('Selected GPU already exceeds memory threshold or free reserve')
    baseline_used=used;baseline_free=free
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
    reason=None;started=time.monotonic();peak=used;owned_peak=0.;util=[]
    telemetry=a.log.with_suffix(a.log.suffix+'.resources.jsonl')
    with a.log.open('w') as log,telemetry.open('x') as resources:
        # Recurrent trajectory lengths vary. This allocator setting completed
        # the v2 round01 same-run recovery without cache growth/guard trips.
        # Set it before importing torch; task/PPO/physical limits are unchanged.
        child=subprocess.Popen(['docker','exec','-e','PYTORCH_ALLOC_CONF=expandable_segments:True',
            '-w',a.workdir,a.container,'/isaac-sim/python.sh','-c',bootstrap,pidfile,token,*command],stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if a.control and not control_enabled(a.control):
                    reason='cancelled_by_user';break
                if a.timeout>0 and time.monotonic()-started>a.timeout: reason='timeout';break
                if FATAL.search(a.log.read_text(errors='replace')): reason='physical_or_python_error';break
                gpu_out=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}',
                    '--query-gpu=memory.used,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                used,total,free,gpu_util=map(float,gpu_out.strip().split(','));peak=max(peak,used);util.append(gpu_util)
                # Attribute memory to this dedicated container, not other card users.
                usage=memory_by_container(a.gpu,[a.container,*a.memory_peer]);owned=usage[a.container]
                owned_peak=max(owned_peak,owned)
                pressure=used/total>=a.memory_fraction or free<a.reserve_mib
                victim=pressure_victim(a.container,usage) if pressure else None
                resources.write(json.dumps(dict(elapsed_s=time.monotonic()-started,
                    card_used_mib=used,card_total_mib=total,card_free_mib=free,card_util_percent=gpu_util,
                    step_container_mib=owned,peer_container_mib={k:v for k,v in usage.items() if k!=a.container},
                    memory_pressure_victim=victim))+'\n');resources.flush()
                if victim==a.container:
                    # The sibling may have released memory between the card
                    # sample and the per-container PID sample. Do not stop the
                    # surviving slot based on the stale high card reading.
                    fresh=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}',
                        '--query-gpu=memory.used,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
                    latest_used,latest_total,latest_free=map(float,fresh.strip().split(','))
                    if latest_used/latest_total>=a.memory_fraction or latest_free<a.reserve_mib:
                        reason='gpu_memory_threshold';break
                time.sleep(2.)
        except BaseException:
            reason='supervisor_interrupted'
            raise
        finally:
            if reason:
                subprocess.run(['docker','exec',a.container,'/isaac-sim/python.sh','-c',kill,pidfile,token],check=False)
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    subprocess.run(['docker','exec',a.container,'/isaac-sim/python.sh','-c',kill.replace('SIGTERM','SIGKILL'),pidfile,token],check=False)
                    child.wait(timeout=15)
            # Native Kit fork workers can outlive a successful launcher and retain
            # file locks. Clean only this token, including after a normal exit.
            subprocess.run(['docker','exec','-w',a.workdir,a.container,'/isaac-sim/python.sh',
                            'tools/cleanup_run.py','--token',token],check=True)
    matches=FATAL.findall(a.log.read_text(errors='replace'))
    artifact=json.loads(a.status.read_text()) if a.status.exists() else {}
    valid=(reason is None and child.returncode==0 and not matches and artifact.get('status') in ('complete','interrupted','passed'))
    result=dict(valid=valid,exit_code=child.returncode,reason=reason,physical_or_python_errors=matches,
                artifact_status=artifact.get('status'),gpu_peak_mib=peak,elapsed_s=time.monotonic()-started,
                gpu_baseline_mib=baseline_used,gpu_baseline_free_mib=baseline_free,step_container_peak_mib=owned_peak,
                card_util_mean_percent=sum(util)/len(util) if util else None,
                memory_fraction=a.memory_fraction,reserve_mib=a.reserve_mib,telemetry=str(telemetry),
                pytorch_alloc_conf='expandable_segments:True')
    result.update(container=a.container,physical_gpu=a.gpu,memory_peers=a.memory_peer,timeout_s=a.timeout)
    a.log.with_suffix(a.log.suffix+'.guard.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    return 0 if valid else 82

if __name__=='__main__': raise SystemExit(main())
