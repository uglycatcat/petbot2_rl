"""Exercise real finite queue with fake processes: no overlapping PPO or reuse after failure."""
import importlib.util,json,pathlib,sys,tempfile
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import finite_waves,launch_cycle

def run_case(fail):
 with tempfile.TemporaryDirectory(dir='/tmp',prefix='petbot-waves-') as tmp:
  root=pathlib.Path(tmp);out=root/'demo_project/runs/_test';out.mkdir(parents=True)
  state=dict(name='test',iterations=1000,train={});clock=[0.];children=[];started=[]
  for i in range(6):
   name=f'run{i}';state['train'][name]=dict(status='queued',gpu=i%3+1,start_reserve_mib=6500,command=['mock',name])
  class Child:
   def __init__(self,cmd,**kwargs):
    self.name=cmd[1] if cmd[0]=='mock' else None;self.pid=10000+len(children);self.start=clock[0];self.done=False
    if self.name:
     gpu=state['train'][self.name]['gpu']
     assert all(c.done or c.name is None or state['train'][c.name]['gpu']!=gpu for c in children)
     started.append(self.name);(out/(self.name+'.log')).write_text('Learning iteration 0')
    children.append(self)
   def poll(self):
    if clock[0]-self.start<10:return None
    self.done=True;code=80 if fail and self.name=='run0' else 0
    (out/(self.name+'.exit.json')).write_text(json.dumps(dict(exit_code=code,guard_stopped=bool(code))))
    path=root/f'demo_project/runs/{self.name}/logs/{self.name}/model_999.pt';path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    return code
  def sleep(t):clock[0]+=t
  with patch.object(finite_waves.subprocess,'Popen',Child),patch.object(finite_waves.time,'sleep',sleep),patch.object(finite_waves.time,'monotonic',lambda:clock[0]),patch.object(launch_cycle,'usage',lambda:{i:(24564,1000,0) for i in range(4)}):
   try:finite_waves.run_waves(out,state,root)
   except RuntimeError:
    assert fail and started==['run0','run1','run2'] and all(state['train'][f'run{i}']['status']=='queued' for i in range(3,6))
   else:assert not fail and len(started)==6 and state['eval_pid'] and all(x['status']=='training_complete' for x in state['train'].values())
 return True
print(json.dumps(dict(passed=run_case(False) and run_case(True),one_ppo_per_gpu=True,failed_round_freezes_queue=True,evaluator_after_all_verified=True,real_jobs=False)))
