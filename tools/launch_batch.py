"""Host-only launch of committed source snapshots into the two dedicated containers."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); args=p.parse_args()
project=Path(__file__).resolve().parents[1]
cfg=json.loads(Path(args.config).read_text())
if subprocess.check_output(['git','status','--porcelain'],cwd=project).strip():
    raise RuntimeError('Commit and push the complete project before launching a formal batch')
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
remote=subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=project,text=True).strip()
if commit!=remote: raise RuntimeError('The training commit is not the pushed upstream commit')
batch=project/'runs'/cfg['name']
if batch.exists(): raise RuntimeError('Batch path exists; inspect existing worker processes instead of relaunching')
source=batch/'source'; source.mkdir(parents=True)
archive=subprocess.check_output(['git','archive',commit],cwd=project)
with tarfile.open(fileobj=io.BytesIO(archive)) as f: f.extractall(source,filter='data')
(batch/'provenance.json').write_text(json.dumps(dict(commit=commit,config=cfg),indent=2)+'\n')
(batch/'control.json').write_text('{"enabled": true}\n')
workspace=project.parent.parent
for job in cfg['jobs']:
    if job['gpu'] not in (2,3): raise ValueError('This project is only authorized on GPU2/3')
    uuid=subprocess.check_output(['nvidia-smi','-i',str(job['gpu']),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    job=dict(job,gpu_uuid=uuid)
    path=batch/f'{job["name"]}.json'; path.write_text(json.dumps(job,indent=2)+'\n')
    # /workspace is the container mount root, not a host path embedded in model/config loading.
    container_source=Path('/workspace')/source.relative_to(workspace)
    container_job=Path('/workspace')/path.relative_to(workspace)
    subprocess.run(['docker','exec','-d','-w',str(container_source),job['container'],
                    '/isaac-sim/python.sh','tools/worker.py','--job',str(container_job)],check=True)
    print(json.dumps(dict(job=job['name'],container=job['container'],commit=commit)),flush=True)
