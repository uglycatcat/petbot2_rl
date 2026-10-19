"""CUDA task contracts: moving success, raw command error and takeover gating."""
import torch
from upright.moving import MovingProgress,slew
from upright.distribution import UprightBetaDistribution
assert torch.cuda.is_available()
device='cuda:0';dt=.06;n=4
p=MovingProgress(n,device,dt)
s=dict(pos=torch.zeros(n,3,device=device),yaw=torch.zeros(n,device=device),forward=torch.full((n,),.2,device=device),vel=torch.zeros(n,6,device=device),q=torch.zeros(n,5,device=device),g=torch.tensor([[0.,0.,-1.]],device=device).repeat(n,1),front_z=torch.full((n,),.18,device=device),rear=torch.ones(n,device=device,dtype=torch.bool),other=torch.zeros(n,device=device,dtype=torch.bool),four=torch.zeros(n,device=device,dtype=torch.bool))
s['pos'][:,2]=.1;s['vel'][:,5]=.5
p.reset(torch.arange(n,device=device),s['pos'],s['yaw']);p.active[:]=True;p.next_command[:]=100;p.raw[:]=torch.tensor([.2,.5],device=device);p.command.copy_(p.raw)
for _ in range(25):
 p.before_step();s['pos'][:,:2]=p.raw_reference;s['yaw'].copy_(p.raw_yaw)
 s['vel'][:,0]=.2*s['yaw'].cos();s['vel'][:,1]=.2*s['yaw'].sin()
 m=p.update(s,s['q'].new_zeros(3))
assert m['held'].all(),'Moving and turning should be accepted without zero-speed/yaw conditions'
# Ignoring operator command cannot be declared successful by matching filtered command.
p.raw[:]=torch.tensor([-.3,-1.],device=device)
m=p.update(s,s['q'].new_zeros(3));assert not m['held'].any()
old=p.command.clone();p.before_step();assert (p.command-old).abs().max()<=2*dt+1e-6
keep=p.since[1].clone();p.reset(torch.tensor([0],device=device),s['pos'],s['yaw']);assert p.since[1]==keep and p.since[0]==0
p.active[:]=False;p.ready[:]=0;p.age[:]=5;p.switch_at[:]=0;s['q'][:,3:]=p.frame[:,None]
s['four'][:]=False
for _ in range(5):m=p.update(s,s['q'].new_zeros(3))
assert not p.active.any()
s['four'][:]=True
for _ in range(3):m=p.update(s,s['q'].new_zeros(3))
assert p.active.all();assert torch.allclose(p.command[:,0],s['forward'])
# Three-channel probability contract and deterministic export agree.
d=UprightBetaDistribution(3,[0.,0.,0.],[.35,.45,.45],shape_cap=128).cuda()
x=d.initial_bias.reshape(1,2,3).repeat(32,1,1).requires_grad_(True);d.update(x)
y=d.sample();lp=d.log_prob(y);d.update(x);assert torch.allclose(lp,d.log_prob(y))
lp.mean().backward();assert torch.isfinite(x.grad).all()
assert torch.allclose(d.mean,torch.jit.script(d.as_deterministic_output_module())(x))
print('PASS moving turning success / raw command accountability / slew / contact gate / partial reset / three-channel Beta')

p.set_command([.1,-.2]);p.next_command[:]=0;p.before_step()
assert torch.allclose(p.raw,torch.tensor([.1,-.2],device=device).expand(n,2))
try:p.set_command([100.,0.]);raise AssertionError('out-of-range accepted')
except ValueError:pass
print('PASS external command persistence and envelope validation')
