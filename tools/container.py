"""Host-side creation of the dedicated step container. No simulator runs on the host."""
import argparse
from pathlib import Path
import subprocess

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='petbot-step-runtime:20261030')
    p.add_argument('--name')
    p.add_argument('--gpu',type=int,choices=(0,1),default=1)
    args=p.parse_args()
    args.name=args.name or f'petbot-step-gpu{args.gpu}'
    project=Path(__file__).resolve().parents[1]; root=project.parents[1]
    subprocess.run(['docker','image','inspect',args.image],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['docker','run','-d','--name',args.name,'--label','petbot.project=step',
        '--gpus',f'device={args.gpu}','--shm-size','4g',
        '-e','ACCEPT_EULA=Y','-e','PRIVACY_CONSENT=Y','-e','OMNI_KIT_ALLOW_ROOT=1',
        '-v',f'{project}:/workspace/demo_project/step_rl_demo',
        '-v',f'{root/"robot_description"}:/workspace/robot_description:ro',
        '-v',f'{root/"demo_project/runs"}:/workspace/runs',
        '-w','/workspace/demo_project/step_rl_demo',args.image,
        'bash','-c','while true; do sleep 3600; done'],check=True)

if __name__=='__main__': main()
