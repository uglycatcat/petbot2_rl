"""Host-only short-run orchestration with actual per-container NVIDIA memory samples."""
import argparse,json,re,subprocess,time
from pathlib import Path
from campaign import container_path,atomic,free_mib
p=argparse.ArgumentParser();p.add_argument('--container',required=True);p.add_argument('--gpu',type=int,required=True)
p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,required=True)
p.add_argument('--iterations',type=int,default=8);a=p.parse_args()
a.output.mkdir(exist_ok=False)
cmd=['docker','exec','-e','PYTHONUNBUFFERED=1','-w',container_path(a.source),a.container,
     '/workspace/isaaclab/isaaclab.sh','-p','train.py','--config',container_path(a.config),
     '--output',container_path(a.output),'--asset-dir',container_path(Path(__file__).resolve().parents[1]/'assets'),
     '--num-envs',str(a.num_envs),'--iterations',str(a.iterations),'--headless','--device','cuda:0']
samples=[];stopped=False
with (a.output/'console.log').open('w') as log:
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    while proc.poll() is None:
        top=subprocess.check_output(['docker','top',a.container,'-eo','pid,args'],text=True)
        pids={l.split()[0] for l in top.splitlines()[1:]}
        apps=subprocess.check_output(['nvidia-smi','--id='+str(a.gpu),'--query-compute-apps=pid,used_gpu_memory',
            '--format=csv,noheader,nounits'],text=True)
        own=sum(int(l.split(',')[1]) for l in apps.splitlines() if l.split(',')[0].strip() in pids)
        free=free_mib(a.gpu);sample={'time':time.time(),'own_compute_mib':own,'card_free_mib':free}
        samples.append(sample)
        with (a.output/'resources.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
        if free<1536 and not stopped:
            pattern='python3 train.py.*'+re.escape(container_path(a.output))+'( |$)'
            subprocess.run(['docker','exec',a.container,'pkill','-USR1','-f',pattern],check=False);stopped=True
        time.sleep(2)
atomic(a.output/'measurement.json',{'returncode':proc.returncode,'num_envs':a.num_envs,'gpu':a.gpu,
    'own_peak_mib':max(s['own_compute_mib'] for s in samples),'minimum_card_free_mib':min(s['card_free_mib'] for s in samples),
    'memory_stop':stopped,'samples':len(samples)})
print((a.output/'measurement.json').read_text())
