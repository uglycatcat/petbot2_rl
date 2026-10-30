"""Regression checks for the batch24 grounded-reward trap."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from upright.scratch_learning import rise_rewards, rear_proximity
device='cuda:0';n=6
s=dict(front_z=torch.full((n,),.045,device=device),rear_fraction=torch.ones(n,device=device),
    rear_z=torch.full((n,2),.030,device=device),
    g=torch.tensor([[0.,0.,-1.]]*n,device=device))
m=dict(phase=torch.tensor([1,1,1,2,3,4],device=device),entered_balance=torch.zeros(n,device=device,dtype=torch.bool),
    failure=torch.zeros(n,device=device,dtype=torch.bool),switched=torch.zeros(n,device=device,dtype=torch.bool),
    transition_track=torch.ones(n,device=device),smooth_reward=torch.ones(n,device=device),
    lift_reward=torch.tensor([0.,0.,0.,.2,.3,.4],device=device))
peak=torch.zeros(n,device=device)
peak,r=rise_rewards(s,m,peak,.02)
assert not r['transition_track'][:3].any() and not r['smooth_reward'][:3].any()
assert (r['transition_track'][3:]==1).all() and (r['smooth_reward'][3:]==1).all()
assert torch.equal(r['lift_reward'][3:],m['lift_reward'][3:])
# A body/hip lean with all four wheels down gives finite exploratory credit;
# dwelling or repeating that lean never earns it again and never means success.
s['g'][0]=torch.tensor([-.8,0.,-.6],device=device);earned=0.
for step in range(500):
    if step%2:s['g'][0,0]=0.
    else:s['g'][0,0]=-.8
    peak,r=rise_rewards(s,m,peak,.02);earned+=float(r['rise_progress_reward'][0])*.02
assert abs(earned-.25)<1e-6 and not r['transition_track'][:3].any()
# Taking over from an existing lean does not count as policy progress.
m['switched'][1]=True;s['g'][1]=torch.tensor([-.8,0.,-.6],device=device)
peak,r=rise_rewards(s,m,peak,.02);assert r['rise_progress_reward'][1]==0
# Real front-wheel lift bridges to the unchanged support/settling criteria.
s['front_z'][0]=.12;peak,r=rise_rewards(s,m,peak,.02)
assert r['rise_progress_reward'][0]>0 and r['transition_track'][0]>.99
m['failure'][2]=True;s['front_z'][2]=.12;peak,r=rise_rewards(s,m,peak,.02)
assert r['rise_progress_reward'][2]==0
assert r['lift_reward'][2]==0 and r['transition_track'][2]==0
# Loss of simultaneous contact impulses must not erase geometrically supported lift.
s['rear_fraction'][0]=0.;s['rear_z'][0]=.0306
_,r=rise_rewards(s,m,peak,.02)
assert r['lift_reward'][0]>.99 and r['rise_contact_cost'][0]==0
# Actual takeoff of either rear wheel, or invalid penetration, earns no support.
s['rear_z'][0,1]=.041
_,r=rise_rewards(s,m,peak,.02)
assert r['lift_reward'][0]==0 and r['rise_contact_cost'][0]==1
heights=torch.tensor([[.030,.030],[.0306,.0306],[.0355,.030],[.04,.03],[.026,.03]],device=device)
assert torch.allclose(rear_proximity(heights),torch.tensor([1.,1.,.5,0.,0.],device=device),atol=1e-6)
print('PASS: supported lift survives contact duty variation; airborne rear wheels rejected; bounded progress and later phases preserved')
