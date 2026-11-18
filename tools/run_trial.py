"""Freeze one published scratch-training trial on a designated project GPU."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tarfile
import time
from campaign import PROJECT, run_job, atomic

parser=argparse.ArgumentParser()
parser.add_argument('--batch',type=Path,required=True)
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--gpu',type=int,choices=[0,1],required=True)
args=parser.parse_args()
if subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT,text=True).strip():
    raise RuntimeError('Publish a clean project before freezing a trial')
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
remote=subprocess.check_output(['git','config','--get','branch.demoproj/step_explore.remote'],cwd=PROJECT,text=True).strip()
published=subprocess.check_output(['git','ls-remote',remote,'refs/heads/demoproj/step_explore'],cwd=PROJECT,text=True).split()[0]
if published!=commit:raise RuntimeError('Trial source has not been published')
top=subprocess.check_output(['docker','top',f'petbot-step-explore-gpu{args.gpu}','-eo','pid,args'],text=True)
if 'python3 train.py' in top:raise RuntimeError('Training container is busy')
config=json.loads(args.config.read_text())
batch=args.batch.resolve();batch.relative_to(PROJECT/'runs')
batch.mkdir();(batch/'source').mkdir();(batch/'assets').mkdir()
with (batch/'source.tar').open('wb') as f:
    subprocess.run(['git','archive',commit],cwd=PROJECT,stdout=f,check=True)
with tarfile.open(batch/'source.tar') as source:source.extractall(batch/'source',filter='data')
for name in ['robot.usda','model.json']:shutil.copy2(PROJECT/'assets'/name,batch/'assets'/name)
atomic(batch/'batch.json',{'commit':commit,'jobs':{args.gpu:config},'from_scratch':True,'created_at':time.time()})
result=run_job(args.gpu,batch,config)
atomic(batch/'review.json',{'results':[result],'reviewed_at':time.time()})
print(json.dumps(result),flush=True)
