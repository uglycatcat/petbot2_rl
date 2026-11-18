"""Resume one resource-interrupted run with the same immutable source/configuration.
Host orchestration only; run_job retains the card-memory guard and independent evaluation.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import time
from campaign import PROJECT, run_job, atomic

parser=argparse.ArgumentParser()
parser.add_argument('--batch',type=Path,required=True)
parser.add_argument('--gpu',type=int,choices=[0,1],required=True)
args=parser.parse_args()
batch=args.batch.resolve()
batch.relative_to(PROJECT/'runs')
out=batch/f'gpu{args.gpu}'
lock=(out/'resume.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
atomic(out/'resume_watchdog.json',{'pid':os.getpid(),'started_at':time.time()})
config=json.loads((out/'config.json').read_text())
result=run_job(args.gpu,batch,config,resume=True)
atomic(out/'resume_result.json',result)
print(json.dumps(result),flush=True)
