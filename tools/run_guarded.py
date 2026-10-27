"""Run an upright job only with GPU memory headroom; stop only the token-owned job on budget breach."""
import argparse, pathlib, subprocess, time, json, uuid, fcntl, os
from job_control import signal_job
from physics_log import PhysicsLogReader, PythonLogReader
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1,2,3],required=True)
p.add_argument('--container',default='petbot-upright-train')
p.add_argument('--entrypoint',default='train.py');p.add_argument('--workdir',default='/workspace/demo_project/upright_rl_demo');p.add_argument('--reserve-mib',type=int,default=5000);p.add_argument('--log',required=True)
a, training_args=p.parse_known_args()
env_counts=[int(training_args[i+1]) for i,x in enumerate(training_args) if x=='--num_envs']+[int(x.split('=',1)[1]) for x in training_args if x.startswith('--num_envs=')]
if any(n>4096 for n in env_counts):
    raise SystemExit('PetBot validated limit is4096 environments per process; larger scenes exceeded the PhysX64K material limit. Use separate guarded processes.')
def usage():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    return {int(i):(int(t),int(u)) for i,t,u in (row.split(',') for row in rows.splitlines())}
total,used=usage()[a.gpu]
if used+a.reserve_mib>total*.78:raise SystemExit(f'Insufficient headroom: used={used}, reserve={a.reserve_mib}, guard={total*.78}')
log=pathlib.Path(a.log);log.parent.mkdir(parents=True,exist_ok=True)
token=uuid.uuid4().hex
command=['docker','exec','-w',a.workdir,'-e',f'CUDA_VISIBLE_DEVICES={a.gpu}','-e',f'PETBOT_JOB_TOKEN={token}','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',a.container,'/isaac-sim/python.sh',a.entrypoint,*training_args]
peak=used;start=time.time();stopped=False;physics_error=None;runtime_error=None
physics_log=PhysicsLogReader(log)
runtime_log=PythonLogReader(log)
with log.open('w') as f:
    child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
    while child.poll() is None:
        physics_error=physics_log.check()
        runtime_error=runtime_log.check()
        if physics_error or runtime_error:
            f.write('\nSimulation validity guard: stopping only this token-owned job.\n');f.flush()
            signal_job(a.container,token,15)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_job(a.container,token,9);child.wait(timeout=15)
            break
        total,used=usage()[a.gpu];peak=max(peak,used)
        if used>=total*.78:
            # Serialize same-user guards: another job may already have freed memory.
            with open(f'/tmp/petbot-upright-guard-{os.getuid()}-{a.gpu}.lock','a') as guard_lock:
                fcntl.flock(guard_lock,fcntl.LOCK_EX)
                total,used=usage()[a.gpu]
                if used>=total*.78:
                    f.write(f'\nMemory guard: GPU{a.gpu} {used}/{total} MiB; stopping only this token-owned job.\n');f.flush()
                    signal_job(a.container,token,15)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        signal_job(a.container,token,9);child.wait(timeout=15)
                    stopped=True;time.sleep(1)
            if stopped:break
        time.sleep(1)
    physics_error=physics_error or physics_log.check()
    runtime_error=runtime_error or runtime_log.check()
    code=81 if physics_error else 82 if runtime_error else 80 if stopped else child.returncode
log.with_suffix('.exit.json').write_text(json.dumps({'job_token':token,'exit_code':code,'gpu':a.gpu,'command':command,'sampled_peak_mib':peak,'total_mib':total,'guard_stopped':stopped,'physics_error':physics_error,'runtime_error':runtime_error,'elapsed_s':time.time()-start}))
raise SystemExit(code)
