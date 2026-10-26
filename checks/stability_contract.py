"""CUDA guard/reward edge cases plus counterexamples for functional scoring."""
import argparse,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from upright.stability import StabilityMonitor,coarse_tracking_score
from upright.continuous import ContinuousProgress
from upright.functional import score_trajectories

p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
assert torch.cuda.is_available();device='cuda:0'
command=torch.tensor([[.05,0.],[-.05,0.],[0.,.15],[0.,-.15],[.04,.1],[0.,0.]],device=device)
assert torch.allclose(coarse_tracking_score(command,command),torch.ones(6,device=device))
assert (coarse_tracking_score(torch.zeros_like(command),command)[:5]<=.25).all()
assert (coarse_tracking_score(-command,command)[:5]<=.25).all()
assert (coarse_tracking_score(command*.8,command)>.99).all()
monitor=StabilityMonitor(3,device,.06);phase=torch.tensor([1,2,3],device=device)
state=dict(front_z=torch.full((3,),.04,device=device),rear_fraction=torch.ones(3,device=device),body_history=torch.zeros(3,dtype=torch.bool,device=device))
zero=torch.zeros(3,2,device=device)
for _ in range(4):assert not monitor.update(phase,state,zero,zero).any()
flags=monitor.update(phase,state,zero,zero)
assert flags[1,0] and flags.sum()==1
monitor.reset(torch.arange(3,device=device));state['front_z'][:]=.18
for _ in range(20):assert not monitor.update(phase,state,zero,zero).any()
state['front_z'][1]=.04
for _ in range(3):assert not monitor.update(phase,state,zero,zero).any()
state['front_z'][1]=.18;assert not monitor.update(phase,state,zero,zero).any()
assert monitor.dwell.sum()==0
progress=ContinuousProgress(1,device,.06,hold_range=(120.,120.))
ids=torch.arange(1,device=device);progress.reset(ids,torch.zeros(1,3,device=device),torch.zeros(1,device=device))
sequence=[]
for _ in range(31):sequence.append(int(progress.mode[0]));progress.sample_commands(ids)
assert set(zip(sequence,sequence[1:]))=={(a,b) for a in range(6) for b in range(6) if a!=b}
rows=[];position=np.zeros(2);heading=0.
commands=[[.05,0.],[-.05,0.],[0.,.15],[0.,-.15],[.04,.10],[0.,0.]]
for index,c in enumerate(commands):
 for step in range(70):
  middle=heading+.03*c[1];position+=.06*c[0]*np.array([np.cos(middle),np.sin(middle)]);heading+=.06*c[1]
  rows.append(dict(t=(len(rows)+1)*.06,raw=[c],valid=[True],phase=[2],pos=[[*position,0.1]],heading=[heading],interval_motion=[c]))
outcomes=dict(complete=[True],failure=[False])
assert score_trajectories(rows,outcomes)['functional_pass']==1
stationary=[dict(r,interval_motion=[[0.,0.]],pos=[[0.,0.,.1]],heading=[0.]) for r in rows]
assert score_trajectories(stationary,outcomes)['functional_pass']==0
assert score_trajectories(rows,dict(complete=[False],failure=[True]))['functional_pass']==0
drift=[dict(r,pos=[list(r['pos'][0])]) for r in rows]
for i in range(350,420):drift[i]['pos'][0][0]+=(i-350)*.006
assert score_trajectories(drift,outcomes)['functional_pass']==0
pathlib.Path(args.output).write_text(json.dumps(dict(passed=True,device=device,reward_counterexamples=True,persistent_guard=True,phase_isolation=True,all_directed_transitions=True,functional_counterexamples=True),indent=2))
print('PASS stability reward, dwell/reset/phase guards, command transitions, functional counterexamples')
