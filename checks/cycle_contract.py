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
p.request_mode(1,-.2);assert (p.mode==0).all() and (p.destination==.2).all(),'Latch landing'
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
