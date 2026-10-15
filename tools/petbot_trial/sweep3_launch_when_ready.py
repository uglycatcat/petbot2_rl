"""One-shot prerequisite wait and launch, never restart a failed training job."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path.cwd().resolve()
if not (root/'default.xml').exists():raise RuntimeError('Run from the manifest workspace')
out=root/'demo_project/runs/_sweep3_precheck'
lock=(out/'launcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (root/'demo_project/runs/_sweep3/status.json').exists():raise RuntimeError('Sweep already has status; inspect before any launch')
required=['train_v2_b.exit','train_v2_c.exit','checkpoint_v2.exit','full_preparation.exit']
start=time.time()
while True:
    for name in required:
        p=out/name
        if p.exists() and p.read_text().strip()!='0':raise RuntimeError(f'Failed prerequisite: {name}')
    if all((out/name).exists() for name in required):break
    if time.time()-start>7200:raise TimeoutError('Prerequisites did not finish within two hours')
    time.sleep(30)
cmd=['docker','exec','petbot-train','/isaac-sim/python.sh','/workspace/demo_env/sweep3_gate.py',
     '--runs-root','/workspace/runs','--write-gate']
with (out/'gate_check.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
print('All prerequisites passed; launching fixed-version sweep',flush=True)
subprocess.run([sys.executable,str(root/'demo_project/demo_env/sweep3_pipeline.py'),'--root',str(root)],check=True)
