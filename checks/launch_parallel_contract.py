"""Run the actual launcher against fake child processes and isolated filesystem."""
import argparse,importlib.util,json,pathlib,sys,tempfile,subprocess
from unittest.mock import patch

p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
source=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(source/'tools'))
spec=importlib.util.spec_from_file_location('tested_launcher',source/'tools/launch_cycle.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
clock=[0.];children=[];started=[];max_pending=[0]
with tempfile.TemporaryDirectory(dir=pathlib.Path(args.output).parent) as directory:
    root=pathlib.Path(directory).resolve();runs=root/'demo_project/runs';repo=root/'demo_project/upright_rl_demo'
    runs.mkdir(parents=True);(repo/'artifacts/petbot_asset').mkdir(parents=True);(runs/'initial.pt').touch()
    launcher.ROOT=root;launcher.RUNS=runs;launcher.SOURCE=repo
    (root/'gate.json').write_text(json.dumps(dict(commit='fixed',passed=True,smokes=[],evaluations=[])))
    variants=[dict(run=f'r{i}',gpu=gpu,envs=2048,reserve_mib=3500,seed=i,initial='demo_project/runs/initial.pt',wheel_slew=480,frame_slew=3,observation_frame='local' if i%2 else 'world') for i,gpu in enumerate([0,1,2,3,0,1])]
    for i,variant in enumerate(variants):variant['rise_speed_reference']='brake' if i%2 else 'capture'
    plan=dict(name='test',commit='fixed',gate='gate.json',variants=variants,container='test',iterations=200,checkpoints=[50],save_interval=50,learning_rate=.0001,envelope=[])
    (root/'plan.json').write_text(json.dumps(plan))
    def output(cmd,**kwargs):
        if cmd[0]=='nvidia-smi':return ''.join(f'{i}, 24564, 1000, 0\n' for i in range(4))
        if 'rev-parse' in cmd:return 'fixed\n'
        return ''
    def run(cmd,**kwargs):
        if 'clone' in cmd:pathlib.Path(cmd[-1]).mkdir(parents=True)
        return subprocess.CompletedProcess(cmd,0)
    class Child:
        def __init__(self,cmd,**kwargs):
            self.pid=800000+len(children);self.returncode=None;self.start=clock[0];self.gpu=None;self.log=None
            if '--gpu' in cmd:
                self.gpu=int(cmd[cmd.index('--gpu')+1]);self.log=pathlib.Path(cmd[cmd.index('--log')+1])
                assert all(c.gpu!=self.gpu or c.log.exists() for c in children),'Same GPU started before earlier PPO'
                started.append((self.gpu,clock[0]));max_pending[0]=max(max_pending[0],1+sum(c.gpu is not None and not c.log.exists() for c in children))
            children.append(self)
        def poll(self):
            if self.log is not None and clock[0]-self.start>=4:self.log.write_text('Learning iteration 0\n')
            return None
    def sleep(seconds):clock[0]+=seconds
    with patch.object(launcher.subprocess,'check_output',output),patch.object(launcher.subprocess,'run',run),patch.object(launcher.subprocess,'Popen',Child),patch.object(launcher.time,'monotonic',lambda:clock[0]),patch.object(launcher.time,'sleep',sleep),patch.object(sys,'argv',['launch_cycle.py',str(root/'plan.json')]):
        launcher.main()
    state=json.loads((runs/'_test/launch_status.json').read_text())
    assert len(state['train'])==6 and all(x['status']=='ppo_started' for x in state['train'].values())
    assert max_pending[0]==4 and len({t for _,t in started[:4]})==1 and started[4][1]>=4
    assert state['eval_pid'] and state['resource_sample_pid']
    for variant in variants:
        command=state['train'][variant['run']]['command']
        assert command[command.index('--observation-frame')+1]==variant['observation_frame']
        assert command[command.index('--rise-speed-reference')+1]==variant['rise_speed_reference']
pathlib.Path(args.output).write_text(json.dumps(dict(passed=True,max_cross_gpu_initializing=max_pending[0],same_gpu_ppo_barrier=True,all_ppo_before_evaluator=True,launched=6,real_jobs=False),indent=2));print('PASS parallel GPU startup and same-GPU PPO barrier')
