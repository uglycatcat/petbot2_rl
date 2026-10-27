"""CUDA state/command checks; complements, never replaces, PhysX and reload."""
import json,pathlib,sys
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.full_range import FullRangeProgress,project_command,effective_request
from upright.fast_feedback import FastYawFeedback

assert torch.cuda.is_available()
device='cuda:0';n=64;ids=torch.arange(n,device=device)
p=FullRangeProgress(n,device,.06,.75,15.,(-.2,.2),42,'pose_feedback_tight',
    hold_range=(8.,12.),tracking_objective='stability_first',handoff_alignment='after',transition_contract='linear_heading')
pos=torch.zeros(n,3,device=device);pos[:,2]=.1;yaw=torch.zeros(n,device=device)
p.reset(ids,pos,yaw)
seen=torch.zeros(n,6,dtype=torch.bool,device=device)
for _ in range(120):
    seen[ids,p.mode]=True
    assert (p.raw[:,0].abs()/.03+p.raw[:,1].abs()*(.1/.06)<=25.00001).all()
    p.sample_commands(ids)
assert seen.all()
p.reset(ids,pos,yaw);p.set_command([0.,15.]);p.phase[:]=2;p.active[:]=True
for _ in range(80):p.before_step()
assert torch.allclose(p.command[:,1],torch.full((n,),15.,device=device))
p.set_command([.75,15.]);assert torch.allclose(p.raw,torch.tensor([.375,7.5],device=device).expand(n,2))

def state(lifted=False):
    return dict(pos=pos.clone(),yaw=yaw.clone(),forward=torch.zeros(n,device=device),
      vel=torch.zeros(n,6,device=device),q=torch.zeros(n,5,device=device),
      g=torch.tensor([-.8,0.,-.6] if lifted else [0.,0.,-1.],device=device).expand(n,3).clone(),
      front_z=torch.full((n,),.18 if lifted else .045,device=device),rear=torch.ones(n,dtype=torch.bool,device=device),
      other=torch.zeros(n,dtype=torch.bool,device=device),four=torch.full((n,),not lifted,device=device),
      rear_fraction=torch.ones(n,device=device),four_fraction=torch.full((n,),0. if lifted else 1.,device=device),
      other_history=torch.zeros(n,dtype=torch.bool,device=device),body_history=torch.zeros(n,dtype=torch.bool,device=device),
      front_vertical_speed=torch.zeros(n,device=device),front_force_peak=torch.zeros(n,device=device),
      front_contact=torch.full((n,),not lifted,device=device),heading_rate=torch.zeros(n,device=device))
fixed=torch.zeros(3,device=device)
p.reset(ids,pos,yaw);p.active[:]=True;p.phase[:]=1;p.external_mode[:]=True
for _ in range(4):m=p.update(state(),fixed)
assert not m['interrupted'].any() and not m['failure'].any()
p.update(state(True),fixed)
assert p.lift_seen.all()
for _ in range(2):m=p.update(state(),fixed)
assert m['interrupted'].all() and m['failure'].all() and not m['completed'].any()
assert (p.phase==4).all() and (p.requested_mode==0).all()

p.reset(ids,pos,yaw);p.active[:]=True;p.phase[:]=2;p.external_mode[:]=True;p.lift_seen[:]=True;p.requested_mode[:]=0;p.destination[:]=0
p.update(state(True),fixed)
for _ in range(24):m=p.update(state(),fixed)
assert m['completed'].all() and not m['interrupted'].any()

p.reset(ids,pos,yaw);p.active[:]=True;p.phase[:]=1;p.command[:]=torch.tensor([.12,0.],device=device)
before=effective_request(p).clone();p.raw[:]=torch.tensor([-.75,15.],device=device)
assert torch.equal(effective_request(p),before)

p.reset(ids,pos,yaw);p.active[:]=True;p.phase[:]=2;p.external_mode[:]=True;p.phase_time[:]=2.;p.raw[:]=0;p.command[:]=0
p.motion[:,0]=.6
for _ in range(15):m=p.update(state(True),fixed)
assert not m['failure'].any(), 'Normal braking must not be classified as falling'

c=FastYawFeedback(n,device,period=.002,kp=16.);c.anti_windup=True;c.half_difference_limit=25.
wheel=torch.full((n,2),25.,device=device);cmd=torch.tensor([0.,15.],device=device).expand(n,2)
for _ in range(100):
    target=c.process(wheel,wheel,wheel,cmd,torch.zeros(n,device=device),torch.zeros(n,device=device),torch.ones(n,device=device,dtype=torch.bool))
assert (target.abs()<=25.).all() and not c.integral.any()
print(json.dumps(dict(passed=True,cuda=True,checks=['joint_feasible_domain','all_six_categories','full_pure_yaw_target','initial_ground_not_interruption','early_return_in_rise','normal_completion','queued_command_isolation','braking_not_fall','saturation_anti_windup'])))
