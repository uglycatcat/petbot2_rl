"""Create this project's isolated GPU0/1 containers. Host orchestration only."""
from pathlib import Path
import argparse
import json
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--gpu',type=int,choices=[0,1],required=True)
a=p.parse_args()
project=Path(__file__).resolve().parents[1]
root=project.parents[1]
name=f'petbot-step-explore-gpu{a.gpu}'
image='nvcr.io/nvidia/isaac-lab:2.3.2'
existing=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
if existing.returncode==0:
    data=json.loads(existing.stdout)[0]
    ids=data['HostConfig']['DeviceRequests'][0]['DeviceIDs']
    if ids!=[str(a.gpu)]:raise RuntimeError('Existing project container has incorrect GPU isolation')
    if not data['State']['Running']:subprocess.run(['docker','start',name],check=True)
else:
    subprocess.run(['docker','run','-d','--name',name,'--gpus',f'device={a.gpu}','--shm-size=4g',
        '-e','ACCEPT_EULA=Y','-e','PRIVACY_CONSENT=Y','-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
        '-v',f'{root}:/workspace/petbot:ro',
        '-v',f'{project}:/workspace/petbot/demo_project/step_explore:rw',
        '--entrypoint','/bin/bash',image,'-lc','exec sleep infinity'],check=True)
subprocess.run(['docker','exec',name,'/isaac-sim/python.sh','-m','pip','install','--no-index',
    '--find-links=/workspace/petbot/demo_project/step_explore/assets/wheels','mujoco==3.3.7'],check=True)
print(name,'ready; root model mount read-only, only project writable')
