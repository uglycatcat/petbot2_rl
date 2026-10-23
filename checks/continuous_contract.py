"""CUDA checks for continuous commands, duration, admission and actor transfer."""
import pathlib
import sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from upright.continuous import ContinuousProgress, joint_tracking_score, yaw_feedforward
from upright.transfer import expand_actor

assert torch.cuda.is_available()
device='cuda:0'
n=12
p=ContinuousProgress(n, device, .06)
ids=torch.arange(n,device=device)
pos=torch.zeros(n,3,device=device)
yaw=torch.zeros(n,device=device)
p.reset(ids,pos,yaw)
seen=torch.zeros(n,6,dtype=torch.bool,device=device)
for _ in range(6):
    seen[ids,p.mode]=True
    p.sample_commands(ids)
assert seen.all(), 'Each environment must visit all six categories'
assert (p.hold_duration>=8).all() and (p.hold_duration<=12).all()

old={'mlp.0.weight':torch.randn(8,245,device=device),'obs_normalizer._mean':torch.randn(1,245,device=device),
     'obs_normalizer._std':torch.rand(1,245,device=device)+.1,'obs_normalizer._var':torch.ones(1,245,device=device)}
new={'mlp.0.weight':torch.randn(8,285,device=device),'obs_normalizer._mean':torch.zeros(1,285,device=device),
     'obs_normalizer._std':torch.ones(1,285,device=device),'obs_normalizer._var':torch.ones(1,285,device=device)}
expanded=expand_actor(old,new,old_width=49,new_width=57)
indices=torch.cat([torch.arange(49,device=device)+57*i for i in range(5)])
x=torch.randn(16,285,device=device)
reference=((x[:,indices]-old['obs_normalizer._mean'])/old['obs_normalizer._std'])@old['mlp.0.weight'].T
actual=((x-expanded['obs_normalizer._mean'])/expanded['obs_normalizer._std'])@expanded['mlp.0.weight'].T
assert torch.allclose(reference,actual,atol=3e-5,rtol=3e-5)

error=torch.tensor([[0.,0.],[0.,10.],[10.,10.]],device=device)
score=joint_tracking_score(error,torch.ones_like(error))
assert score[0]==1 and 0<score[2]<score[1]<.3
u=torch.linspace(-.3,.3,11,device=device)
assert torch.allclose(yaw_feedforward(-u),-yaw_feedforward(u))
assert yaw_feedforward(torch.zeros(1,device=device))==0

s=dict(pos=torch.tensor([[0.,0.,.1]],device=device).repeat(n,1),yaw=yaw,forward=torch.zeros(n,device=device),
       vel=torch.zeros(n,6,device=device),q=torch.zeros(n,5,device=device),g=torch.tensor([[-.8,0.,-.6]],device=device).repeat(n,1),
       front_z=torch.full((n,),.18,device=device),rear=torch.ones(n,dtype=torch.bool,device=device),other=torch.zeros(n,dtype=torch.bool,device=device),
       four=torch.ones(n,dtype=torch.bool,device=device),rear_fraction=torch.ones(n,device=device),four_fraction=torch.zeros(n,device=device),
       other_history=torch.zeros(n,dtype=torch.bool,device=device),body_history=torch.zeros(n,dtype=torch.bool,device=device),
       front_vertical_speed=torch.zeros(n,device=device),front_force_peak=torch.zeros(n,device=device),front_contact=torch.zeros(n,dtype=torch.bool,device=device),
       heading_rate=torch.zeros(n,device=device))
p.reset(ids,pos,yaw);p.external[:]=True;p.raw[:]=0;p.raw[:,1]=.15;p.command.copy_(p.raw);p.frame[:]=0;p.age[:]=5;p.switch_at[:]=2
for _ in range(4):p.update(s,torch.zeros(3,device=device))
assert not p.active.any(), 'Stationary yaw state cannot pass a nonzero turn admission'
p.motion[:,1]=.15
for _ in range(4):p.update(s,torch.zeros(3,device=device))
assert p.active.all()
p.phase[:]=2;p.since[:]=15;p.phase_time[:]=1;p.requested_mode[:]=1;p.external_mode[:]=False;p.hold_duration[:]=8
p.update(s,torch.zeros(3,device=device));assert (p.phase==2).all(), 'Old takeover clock must not trigger landing'
p.phase_time[:]=8;p.update(s,torch.zeros(3,device=device));assert (p.phase==3).all()
print('PASS mixed categories, 49-to-57 transfer, joint score, measured admission, hold clock')
