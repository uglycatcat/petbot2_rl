"""Exercise the real scheduler with simulated children; no GPU jobs or notifications."""
import argparse,json,os,pathlib,runpy,subprocess,sys,tempfile,time
from unittest.mock import patch

p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--output',required=True);args=p.parse_args()
source=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(source/'tools'))
evidence=json.loads(pathlib.Path(args.result).read_text())
assert evidence['count']==64 and evidence['unfinished']==0
active={};peak=0;counter=0;attempts=[]
with tempfile.TemporaryDirectory(prefix='queue_check_',dir=pathlib.Path(args.output).parent) as temporary:
    root=pathlib.Path(temporary);out=root/'demo_project/runs/_queue_test';out.mkdir(parents=True)
    jobs={}
    for name in ['a','b','c']:
        f=root/f'demo_project/runs/{name}/logs/{name}/model_0.pt';f.parent.mkdir(parents=True);f.write_bytes(b'test-only')
        os.utime(f,(time.time()-60,time.time()-60));jobs[name]=dict(envelope=[])
    launch=dict(train=jobs,checkpoints=[0],iterations=1,container='test',evaluation_envelopes={},eval_parallelism=2,eval_reserve_mib=4000)
    (out/'launch_status.json').write_text(json.dumps(launch))
    class Child:
        def __init__(self,cmd):
            global peak,counter
            gpu=int(cmd[cmd.index('--gpu')+1]);assert gpu not in active
            assert gpu!=1,'Resource guard must exclude occupied GPU1'
            counter+=1;self.pid=900000+counter;self.gpu=gpu;self.polls=0
            active[gpu]=self;peak=max(peak,len(active));assert len(active)<=2
            self.folder=root/'demo_project/runs'/pathlib.Path(cmd[cmd.index('--log_dir')+1]).relative_to('/workspace/runs')
            attempts.append(gpu)
        def poll(self):
            self.polls+=1
            if self.polls<2:return None
            self.folder.mkdir(parents=True,exist_ok=True)
            (self.folder/'evaluation.json').write_text(json.dumps(evidence))
            # Simulated jobs finish in the same filesystem clock tick; stamp their synthetic completion explicitly.
            os.utime(self.folder/'evaluation.json',(time.time()+1,time.time()+1))
            active.pop(self.gpu,None);return 0
    with patch.object(subprocess,'Popen',Child),patch.object(subprocess,'check_output',return_value='0, 24564, 18000\n1, 24564, 17000\n2, 24564, 1000\n3, 24564, 1000\n'),patch.object(time,'sleep',lambda seconds:None),patch.object(sys,'argv',['evaluate_cycle.py','--root',str(root),'--name','queue_test']):
        runpy.run_path(str(source/'tools/evaluate_cycle.py'),run_name='__main__')
    final=json.loads((out/'eval_finished.json').read_text());state=json.loads((out/'eval_status.json').read_text())
    assert final['complete']==3 and final['failed']==0 and final['expected']==3,(final,{k:{a:b for a,b in v.items() if a in ('status','validation_error')} for k,v in state.items()})
    assert peak==2 and len(attempts)==3 and not active
    assert len({v['pid'] for v in state.values()})==3
pathlib.Path(args.output).write_text(json.dumps(dict(passed=True,peak_parallelism=peak,launched=3,per_gpu_exclusive=True,occupied_gpu_excluded=True,notifications=False),indent=2))
print('PASS bounded concurrent evaluation, no duplicate launch, memory exclusion, validated completion')
