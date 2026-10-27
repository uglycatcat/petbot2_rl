"""One synchronized PPO job on several GPUs, with per-card memory/physics guards."""
import argparse,json,pathlib,subprocess,time,uuid
from job_control import signal_job
from physics_log import PhysicsLogReader

p=argparse.ArgumentParser()
p.add_argument('--gpus',required=True);p.add_argument('--reserve-mib',type=int,default=5500)
p.add_argument('--container',default='petbot-upright-night');p.add_argument('--workdir',required=True)
p.add_argument('--log',required=True)
a,extra=p.parse_known_args();gpus=[int(x) for x in a.gpus.split(',')]
assert len(gpus)>1 and len(set(gpus))==len(gpus) and set(gpus)<={0,1,2,3}
counts=[int(extra[i+1]) for i,x in enumerate(extra) if x=='--num_envs']
assert len(counts)==1 and 1<=counts[0]<=4096
def usage():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    return {int(i):(int(t),int(u)) for i,t,u in (row.split(',') for row in raw.splitlines())}
for g in gpus:
    total,used=usage()[g]
    if used+a.reserve_mib>=total*.78:raise SystemExit(f'Insufficient GPU{g} headroom')
log=pathlib.Path(a.log);log.parent.mkdir(parents=True,exist_ok=True)
if log.exists():raise FileExistsError(log)
token=uuid.uuid4().hex
command=['docker','exec','-w',a.workdir,'-e','CUDA_VISIBLE_DEVICES='+a.gpus,
    '-e','PETBOT_JOB_TOKEN='+token,'-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',
    a.container,'/isaac-sim/python.sh','-m','torch.distributed.run','--standalone',
    '--nproc_per_node='+str(len(gpus)),'train_cycle.py','--distributed',*extra]
start=time.time();peak={g:0 for g in gpus};error=None;guard=False
physics=PhysicsLogReader(log)
with log.open('w') as out:
    child=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT)
    while child.poll() is None:
        memory=usage();error=physics.check()
        for g in gpus:
            total,used=memory[g];peak[g]=max(peak[g],used)
            guard|=used>=total*.78
        if error or guard:
            signal_job(a.container,token,15)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:signal_job(a.container,token,9);child.wait(timeout=15)
            break
        time.sleep(1)
error=error or physics.check()
code=81 if error else 80 if guard else child.returncode
log.with_suffix('.exit.json').write_text(json.dumps(dict(exit_code=code,job_token=token,
    command=command,gpus=gpus,world_size=len(gpus),local_envs=counts[0],total_envs=counts[0]*len(gpus),
    elapsed_s=time.time()-start,guard_stopped=guard,physics_error=error,sampled_peak_mib=peak),indent=2)+'\n')
raise SystemExit(code)
