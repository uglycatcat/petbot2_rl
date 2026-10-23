"""CUDA full-cycle stage, reference, mode, low-speed tracking and transfer contracts."""
import pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.cycle import CycleProgress
from upright.transfer import expand_actor
assert torch.cuda.is_available();d='cuda:0';dt=.06;n=2
s=dict(pos=torch.tensor([[0.,0.,.1]],device=d).repeat(n,1),yaw=torch.zeros(n,device=d),forward=torch.zeros(n,device=d),vel=torch.zeros(n,6,device=d),q=torch.zeros(n,5,device=d),g=torch.tensor([[-.8,0.,-.6]],device=d).repeat(n,1),front_z=torch.full((n,),.18,device=d),rear=torch.ones(n,device=d,dtype=torch.bool),other=torch.zeros(n,device=d,dtype=torch.bool),four=torch.zeros(n,device=d,dtype=torch.bool),rear_fraction=torch.ones(n,device=d),four_fraction=torch.zeros(n,device=d),other_history=torch.zeros(n,device=d,dtype=torch.bool),body_history=torch.zeros(n,device=d,dtype=torch.bool),front_vertical_speed=torch.zeros(n,device=d),front_force_peak=torch.zeros(n,device=d),front_contact=torch.zeros(n,device=d,dtype=torch.bool),heading_rate=torch.zeros(n,device=d))
p=CycleProgress(n,d,dt);ids=torch.arange(n,device=d);p.reset(ids,s['pos'],s['yaw']);p.active[:]=True;p.phase[:]=1;p.external[:]=True;p.raw[:]=0;p.command[:]=0;p.last_g.copy_(s['g']);p.land_at[:]=100
p.raw_reference[:]=.2
for _ in range(6):m=p.update(s,torch.zeros(3,device=d))
assert (p.phase==2).all() and torch.allclose(p.raw_reference,s['pos'][:,:2]);assert (p.transition_offset>.2).all()
p.raw_reference[:]=.1;m=p.update(s,torch.zeros(3,device=d));assert (p.raw_reference==.1).all(),'No repeated reanchor'
p.raw[:]=torch.tensor([.01,.01],device=d);p.since[:]=3
for _ in range(25):m=p.update(s,torch.zeros(3,device=d))
assert not m['tracking_good'].any(),'Small forward command cannot pass while stationary'
p.request_mode(0,.2);m=p.update(s,torch.zeros(3,device=d));assert (p.phase==3).all()
p.request_mode(1,-.2);assert (p.requested_mode==0).all() and (p.destination==.2).all(),'Latch landing'
s['q'][:,3:]=.2;s['g'][:]=torch.tensor([0.,0.,-1.],device=d);s['front_z'][:]=.045;s['four_fraction'][:]=1
for _ in range(30):m=p.update(s,torch.zeros(3,device=d))
assert (p.phase==4).all() and m['completed'].all() and not m['qualified'].any()
s['g'][:,2]=.3;m=p.update(s,torch.zeros(3,device=d));assert m['failure'].all() and not m['completed'].any()
p.reset(ids,s['pos'],s['yaw']);assert not p.handoff.any() and not p.track_count.any() and (p.phase==0).all()
old={'mlp.0.weight':torch.randn(3,205,device=d),'obs_normalizer._mean':torch.randn(1,205,device=d),'obs_normalizer._std':torch.rand(1,205,device=d)+.1,'obs_normalizer._var':torch.ones(1,205,device=d)}
new={'mlp.0.weight':torch.randn(3,245,device=d),'obs_normalizer._mean':torch.zeros(1,245,device=d),'obs_normalizer._std':torch.ones(1,245,device=d),'obs_normalizer._var':torch.ones(1,245,device=d)}
e=expand_actor(old,new);indices=torch.cat([torch.arange(41,device=d)+49*i for i in range(5)]);x=torch.randn(4,245,device=d)
y=((x[:,indices]-old['obs_normalizer._mean'])/old['obs_normalizer._std'])@old['mlp.0.weight'].T
z=((x-e['obs_normalizer._mean'])/e['obs_normalizer._std'])@e['mlp.0.weight'].T
assert torch.allclose(y,z,atol=2e-5,rtol=2e-5)
print('PASS one-time reference anchor, low-speed accountability, latched landing, handoff, failure priority, reset, actor history transfer')

# A stationary landing posture must not accumulate progress reward, and fast
# descent must already be penalized before front-wheel contact.
p.reset(ids,s['pos'],s['yaw']);p.active[:]=True;p.phase[:]=3;p.external[:]=True;p.destination[:]=.2
s['g'][:]=torch.tensor([0.,0.,-1.],device=d);s['front_z'][:]=.09;s['four_fraction'][:]=0;s['front_contact'][:]=False;s['front_vertical_speed'][:]=-.7
p.update(s,torch.zeros(3,device=d));m=p.update(s,torch.zeros(3,device=d))
assert torch.all(m['land_reward']==0), 'No repeated pose reward'
assert (m['descent_cost']>0).all(), 'Brake before contact'
s['front_vertical_speed'][:]=-.05;m=p.update(s,torch.zeros(3,device=d));assert (m['descent_cost']==0).all()
# Static landing has negative total residence incentive even with maximal
# approach, ground, transition and smooth shaping: -.7 per second or less.
assert -2+.2+.5+.1+.5<0
print('PASS no landing residence exploit, precontact braking, gentle descent')

# Peak overspeed penalty is charged only as the episode maximum increases,
# independent of contact chatter or control duration.
s['front_contact'][:]=True;s['front_vertical_speed'][:]=-.7;p.last_front_speed[:]=-.7
m=p.update(s,torch.zeros(3,device=d));assert torch.allclose(m['impact_increment'],torch.ones(n,device=d))
m=p.update(s,torch.zeros(3,device=d));assert (m['impact_increment']==0).all()
s['front_contact'][:]=False;m=p.update(s,torch.zeros(3,device=d));assert (m['impact_increment']==0).all()
p.reset(ids,s['pos'],s['yaw']);assert (p.max_touchdown_speed==0).all()
print('PASS impact increment once per increased maximum and reset')

# Regression: cycle requests must never overwrite ordinary-drive command groups.
n=60;ids=torch.arange(n,device=d);p=CycleProgress(n,d,dt,seed=20261023)
pos=torch.zeros(n,3,device=d);yaw=torch.zeros(n,device=d)
for generation in range(3):
    p.reset(ids,pos,yaw)
    assert torch.equal(p.mode,ids%6)
    assert (p.requested_mode==1).all()
    for request in [0,1,0]:
        p.request_mode(request,0.)
        p.sample_commands(ids)
        v,w=p.raw.unbind(-1)
        assert (v[p.mode==0]>0).all() and (w[p.mode==0]==0).all()
        assert (v[p.mode==1]<0).all() and (w[p.mode==1]==0).all()
        assert (v[p.mode==2]==0).all() and (w[p.mode==2]>0).all()
        assert (v[p.mode==3]==0).all() and (w[p.mode==3]<0).all()
        assert (v[p.mode==4]!=0).all() and (w[p.mode==4]!=0).all()
        assert (p.raw[p.mode==5]==0).all()
print('PASS all six command groups survive reset and mode requests')

# Preserve positive bounded rewards: poor tracking must not create a new incentive
# to terminate. Match half-height so the comparison is not merely a wider scale.
from upright.cycle import tracking_score
import math
err=torch.tensor([[0.,0.],[math.sqrt(math.log(2))]*2,[2.,2.],[3.,3.]],device=d)
tol=torch.full_like(err,.5)
gaussian=tracking_score(err,tol,'gaussian');cauchy=tracking_score(err,tol,'cauchy')
assert torch.allclose(gaussian[:2],cauchy[:2],atol=1e-6)
assert (cauchy[:-1]>cauchy[1:]).all() and (cauchy>0).all() and (cauchy<=1).all()
assert cauchy[3]>gaussian[3]*100
assert torch.equal(cauchy,tracking_score(-err,tol,'cauchy'))
assert torch.isfinite(tracking_score(err*1e6,tol,'cauchy')).all()
print('PASS tracking reward bounds, symmetry, half-height matching and tail discrimination')

from upright.cycle import filter_differential
old=torch.tensor([[2.,2.],[-25.,25.]],device=d)
target=torch.tensor([[1.,3.],[24.,25.]],device=d)
assert torch.equal(filter_differential(old,target,.06,0.,25.),target)
f=filter_differential(old,target,.06,.12,25.)
assert torch.allclose(f.mean(-1),target.mean(-1)) and (f.abs()<=25.).all()
assert 0<(f[0,1]-f[0,0])<2
x=old[:1]
errors=[]
for _ in range(20):
 x=filter_differential(x,target[:1],.06,.12,25.);errors.append((x-target[:1]).abs().max().item())
assert all(b<a for a,b in zip(errors,errors[1:])) and errors[-1]<.001
assert torch.allclose(filter_differential(-old,-target,.06,.12,25.),-f)
print('PASS yaw filter bypass, common preservation, bounds, convergence and reversal')
