"""Run an upright job only with GPU memory headroom; stop only the token-owned job on budget breach."""
import argparse, pathlib, subprocess, time, json, uuid
from job_control import signal_job
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1,2,3],required=True)
p.add_argument('--container',default='petbot-upright-train')
p.add_argument('--entrypoint',default='train.py');p.add_argument('--workdir',default='/workspace/demo_project/upright_rl_demo');p.add_argument('--reserve-mib',type=int,default=5000);p.add_argument('--log',required=True)
a, training_args=p.parse_known_args()
def usage():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    return {int(i):(int(t),int(u)) for i,t,u in (row.split(',') for row in rows.splitlines())}
total,used=usage()[a.gpu]
if used+a.reserve_mib>total*.78:raise SystemExit(f'Insufficient headroom: used={used}, reserve={a.reserve_mib}, guard={total*.78}')
log=pathlib.Path(a.log);log.parent.mkdir(parents=True,exist_ok=True)
token=uuid.uuid4().hex
command=['docker','exec','-w',a.workdir,'-e',f'CUDA_VISIBLE_DEVICES={a.gpu}','-e',f'PETBOT_JOB_TOKEN={token}','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',a.container,'/isaac-sim/python.sh',a.entrypoint,*training_args]
peak=used;start=time.time();stopped=False
with log.open('w') as f:
    child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
    while child.poll() is None:
        total,used=usage()[a.gpu];peak=max(peak,used)
        if used>=total*.78:
            f.write(f'\nMemory guard: GPU{a.gpu} {used}/{total} MiB; stopping only this token-owned job.\n');f.flush()
            signal_job(a.container,token,15)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_job(a.container,token,9);child.wait(timeout=15)
            stopped=True;break
        time.sleep(1)
    code=80 if stopped else child.returncode
log.with_suffix('.exit.json').write_text(json.dumps({'job_token':token,'exit_code':code,'gpu':a.gpu,'command':command,'sampled_peak_mib':peak,'total_mib':total,'guard_stopped':stopped,'elapsed_s':time.time()-start}))
raise SystemExit(code)
