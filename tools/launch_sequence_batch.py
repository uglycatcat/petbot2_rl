"""Host orchestration of frozen, scratch PhysX sequence searches in owned containers."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time


def command(*args):return subprocess.check_output(args,text=True).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/sequence_physx_round01.json');p.add_argument('--launch',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/args.config).read_text());jobs=config['jobs']
    if config['backend']!='physx' or any(j['gpu'] not in (2,3) for j in jobs):raise ValueError('Only the authorized PhysX GPU2/3 deployment is supported')
    if len({j['container'] for j in jobs})!=len(jobs):raise ValueError('Each job requires its own container')
    free={int(row.split(',')[0]):int(row.split(',')[1]) for row in command('nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits').splitlines()}
    for gpu in {j['gpu'] for j in jobs}:
        need=sum(j['gpu']==gpu for j in jobs)*config['expected_peak_mib_per_job']+config['reserve_mib_per_card']
        if free[gpu]<need:raise RuntimeError(f'GPU{gpu}: {free[gpu]} MiB free, {need} MiB required')
    for job in jobs:
        if not job['container'].startswith('petbot-recover-explore-'):raise ValueError('Not a recover-owned container')
        processes=command('docker','top',job['container'],'-eo','pid,args')
        if any(name in processes for name in ('train.py','evaluate.py','evaluate_mujoco.py','search_sequences','replay_targets_physx.py')):
            raise RuntimeError(f"Container {job['container']} still has task work")
        runtime=json.loads(command('docker','inspect',job['container']))[0]
        devices=runtime['HostConfig']['DeviceRequests'][0]['DeviceIDs']
        uuid=command('nvidia-smi',f"--id={job['gpu']}",'--query-gpu=uuid','--format=csv,noheader')
        if devices not in ([str(job['gpu'])],[uuid]):raise ValueError('Container GPU assignment differs')
    sha=command('git','-C',str(root),'rev-parse','HEAD')
    if command('git','-C',str(root),'status','--porcelain'):raise RuntimeError('Commit the complete change before launching')
    if sha!=command('git','-C',str(root),'rev-parse','ours/demoproj/recover_explore'):raise RuntimeError('Push the frozen source first')
    batch=root/'runs'/config['batch']
    if batch.exists():raise ValueError('Never overwrite or resume a completed batch')
    print(json.dumps(dict(commit=sha,batch=str(batch),jobs=jobs,launch=args.launch)),flush=True)
    if not args.launch:return
    source=batch/'source';source.mkdir(parents=True)
    archive=subprocess.check_output(['git','-C',str(root),'archive','--format=tar',sha])
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:t.extractall(source,filter='data')
    config.update(commit=sha,archive_sha256=hashlib.sha256(archive).hexdigest(),started_at=time.time())
    (batch/'batch.json').write_text(json.dumps(config,indent=2)+'\n');(batch/'control.json').write_text('{"enabled": true}\n')
    runtime_root=Path('/workspace/demo_project/recover_explore')/'runs'/config['batch']
    for job in jobs:
        argv=['/isaac-sim/python.sh','search_sequences_physx.py','--output','../'+job['name'],
              '--seed',str(job['seed']),'--wheel-mode',job['wheel_mode'],'--population-per-pose',str(config['population_per_pose']),
              '--generations',str(config['generations']),'--knots',str(config['knots']),'--threads',str(config['threads']),'--headless']
        shell=shlex.join(argv)+' > '+shlex.quote('../'+job['name']+'.log')+' 2>&1'
        subprocess.run(['docker','exec','-d','-w',str(runtime_root/'source'),job['container'],'bash','-c',shell],check=True)
        print('Started '+job['name'],flush=True)


if __name__=='__main__':main()
