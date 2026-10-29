"""CUDA sampler and curriculum contract checks; not a PhysX acceptance test."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from upright.scratch import ScratchProgress
from upright.symmetry import mirror_action,mirror_observation

def make(evaluating=False):
    return ScratchProgress(256,'cuda:0',.02,.75,15.,(-1.,1.2),60,
        'pose_feedback_tight',hold_range=(12.,30.),tracking_objective='stability_first',
        handoff_alignment='after',transition_contract='linear_heading',
        mirror_training=True,adaptive_curriculum=True,evaluating=evaluating)

p=make();ids=torch.arange(p.n,device=p.device);pos=torch.zeros(p.n,3,device=p.device);yaw=pos[:,0]
p.reset(ids,pos,yaw)
assert not p.raw.any() and not p.frame.any() and (p.hold_duration==4).all()
p.bootstrap_stage=1;p.reset(ids,pos,yaw)
assert (p.raw.abs()<=p.raw.new_tensor([.05,.15])+1e-6).all() and p.frame.abs().max()<=.1
p.bootstrap_stage=2;p.split_levels=[6,7,7,8,8];p.reset(ids,pos,yaw);p.phase[:]=2
values=[]
for i in range(48):
    p.sample_commands(ids);values.append(p.request_target.clone())
    assert (p.request_target[:,0].abs()+.05*p.request_target[:,1].abs()<=.75001).all()
    assert torch.allclose(p.request_target[::2,0],p.request_target[1::2,0])
    assert torch.allclose(p.request_target[::2,1],-p.request_target[1::2,1])
values=torch.stack(values);straight=(p.cohort==5)|(p.cohort==6);v=values[:,straight,0]
assert (v==0).any() and (v==.75).any() and (v==-.75).any()
assert ((v.abs()>0)&(v.abs()<.05)).any() and (v.unique().numel()>100)
state=p.curriculum_state_dict();q=make();q.load_curriculum_state_dict(state)
assert q.split_levels==p.split_levels and q.bootstrap_stage==p.bootstrap_stage
assert torch.equal(q.generation,p.generation)
q.reset(ids,pos,yaw);q.split_levels[2]=0
side=torch.cat((torch.zeros(32,device=p.device),torch.ones(32,device=p.device))).long()
level=torch.zeros_like(side);success=torch.ones_like(side,dtype=torch.bool)
q._record(2,side,level,success);assert q.split_levels[2]==0
q._record(2,side,level,success);assert q.split_levels[2]==1
q._record(2,side,level+1,~success);assert q.split_levels[2]==0
ev=make(True);ev.reset(ids,pos,yaw)
assert not ev.split_enabled and ev.frame.abs().max()>.5 and ev.raw.abs().max()>.5
obs=torch.randn(256,290,device=p.device);action=torch.randn(256,3,device=p.device)
assert torch.equal(mirror_observation(mirror_observation(obs)),obs)
assert torch.equal(mirror_action(mirror_action(action)),action)
print(json.dumps(dict(passed=True,simulator=False,continuous_straight_values=v.unique().numel(),
    paired_commands=True,bootstrap=True,curriculum_advance_and_regress=True,state_reload=True)))
