"""Stop only container processes carrying this launcher's unguessable job token."""
import subprocess

def signal_job(container, token, signal):
    if len(token)!=32 or any(c not in '0123456789abcdef' for c in token):
        raise ValueError('Invalid job token')
    script='''import pathlib,os,signal,sys
needle=('PETBOT_JOB_TOKEN='+sys.argv[1]).encode()
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if needle in (p/'environ').read_bytes().split(b'\\0'):os.kill(int(p.name),int(sys.argv[2]))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
'''
    subprocess.run(['docker','exec',container,'/isaac-sim/kit/python/bin/python3','-c',script,token,str(signal)],check=True)
