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

# A neighbour reset must not alter this environment's future command sequence.
a=MovingProgress(n,device,dt,seed=123);b=MovingProgress(n,device,dt,seed=123)
ids=torch.arange(n,device=device)
a.reset(ids,s['pos'],s['yaw']);b.reset(ids,s['pos'],s['yaw'])
b.reset(torch.tensor([0],device=device),s['pos'],s['yaw'])
rest=torch.tensor([1,2,3],device=device);a.sample_commands(rest);b.sample_commands(rest)
assert torch.equal(a.raw[rest],b.raw[rest]) and torch.equal(a.next_command[rest],b.next_command[rest])
print('PASS independent per-environment evaluation command streams')

# Common acceleration is exactly symmetric, while yaw remains independent.
from upright.moving import wheel_targets
x=torch.tensor([[0.,.4,0.],[0.,0.,1.],[0.,1.,1.]],device=device)
w=wheel_targets(x,25.,'common_diff')
assert torch.equal(w[0],torch.tensor([10.,10.],device=device))
assert torch.equal(w[1],torch.tensor([-5.,5.],device=device))
assert w.abs().max()<=25
assert torch.equal(wheel_targets(x,25.,'independent'),x[:,1:]*25)
# Dense credit below full lift grows with height; success remains unchanged.
p=MovingProgress(n,device,dt);p.active[:]=True;p.since[:]=3
s['four'][:]=False;s['other'][:]=False;s['rear'][:]=True;s['front_z'][:]=.06
low=p.update(s,s['q'].new_zeros(3))['prepare'].clone()
s['front_z'][:]=.10
m=p.update(s,s['q'].new_zeros(3))
assert (m['prepare']>low).all() and not m['support'].any() and not m['held'].any()
assert torch.allclose(m['transition_scale'],torch.full((n,),.1,device=device))
print('PASS common/differential action bounds and dense partial-lift credit without relaxed success')

# Reward feedback corrects accumulated pose error, with rotational invariance.
from upright.moving import pose_velocity_target
pos=torch.zeros(2,2,device=device);yaw=torch.tensor([0.,torch.pi/2],device=device)
ref=torch.tensor([[.2,.0],[0.,.2]],device=device);cmd=torch.tensor([[.1,.2],[.1,.2]],device=device)
out=pose_velocity_target(pos,yaw,ref,yaw,cmd)
assert torch.allclose(out,torch.tensor([[.2,.2],[.2,.2]],device=device),atol=1e-6)
ref=torch.tensor([[0.,.2],[0.,.2]],device=device);yaw.zero_();cmd[:,0]=torch.tensor([.1,-.1],device=device);cmd[:,1]=0
out=pose_velocity_target(pos,yaw,ref,yaw,cmd)
assert out[0,1]>0 and out[1,1]<0
assert ((out-cmd).abs()<=torch.tensor([.1,.25],device=device)+1e-6).all()
assert torch.equal(pose_velocity_target(pos,yaw,pos,yaw,cmd),cmd)
print('PASS bounded pose feedback, rotation invariance, reverse steering and zero-error identity')

# Heading derivative must match independent finite world-axis rotation, including tilt.
from upright.moving import projected_heading_rate,heading_feedback_rewards
axis=torch.tensor([[1.,0.,0.],[.3,.4,.8660254],[.6,-.2,.7745967]],device=device,dtype=torch.float64)
omega=torch.tensor([[0.,0.,.4],[.2,-.4,.3],[-.3,.2,-.5]],device=device,dtype=torch.float64)
eps=1e-6;unit=omega/omega.norm(dim=-1,keepdim=True);angle=omega.norm(dim=-1,keepdim=True)*eps
rot=axis*angle.cos()+torch.cross(unit,axis,dim=-1)*angle.sin()+unit*(unit*axis).sum(-1,keepdim=True)*(1-angle.cos())
finite=(torch.atan2(rot[:,1],rot[:,0])-torch.atan2(axis[:,1],axis[:,0]))/eps
assert torch.allclose(projected_heading_rate(axis,omega),finite,atol=1e-6)
assert (projected_heading_rate(axis,omega)[1:]-omega[1:,2]).abs().min()>.05
bad=torch.tensor([.5],device=device);zero=bad*0
good_turn,_=heading_feedback_rewards(bad,zero,bad,zero)
bad_turn,_=heading_feedback_rewards(bad,bad,bad,zero)
assert good_turn>bad_turn+.7,'Translation error must not erase the turn reward'
tr,pa=heading_feedback_rewards(zero,zero,zero,zero)
assert tr==1 and pa==1
print('PASS tilted heading-rate finite difference and independently bounded turning reward')

# Sustain recipe: turn share, dense heading-window credit, and its boundedness.
m=MovingProgress(n,device,.06,turn_share=.5)
assert m.mode.min()>=0 and m.mode.max()<=5
share=float(((m.mode==2)|(m.mode==3)).float().mean())
assert .4<=share<=.6,f'pure-turn share {share}'
assert float(((MovingProgress(n,device,.06).mode==2)|(MovingProgress(n,device,.06).mode==3)).float().mean())==0.

p=MovingProgress(n,device,dt,feedback_mode='sustain')
p.active[:]=True;p.since[:]=3
s2=dict(s);s2['four']=torch.zeros(n,device=device,dtype=torch.bool)
s2['rear']=torch.ones(n,device=device,dtype=torch.bool);s2['other']=torch.zeros(n,device=device,dtype=torch.bool)
s2['front_z']=torch.full((n,),.18,device=device);s2['g']=torch.tensor([[0.,0.,-1.]],device=device).repeat(n,1)
s2['pos']=torch.zeros(n,3,device=device);s2['yaw']=torch.zeros(n,device=device);s2['vel']=torch.zeros(n,6,device=device)
s2['forward']=torch.zeros(n,device=device);s2['q']=torch.zeros(n,5,device=device);s2['heading_rate']=torch.zeros(n,device=device)
p.raw[:]=0.;p.command[:]=0.;p.reference[:]=0.;p.raw_reference[:]=0.;p.raw_yaw[:]=0.;p.yaw[:]=0.
run=[]
for _ in range(20):
    r=p.update(s2,s2['q'].new_zeros(3));run.append(float(r['heading_sustain'][0]))
assert run[0]<run[5]<run[-1] and max(run)<=1.0001,'sustain credit must ramp with the compliant run'
# A heading violation resets the run instead of paying for stale compliance.
s2['yaw'][:]=.5
r=p.update(s2,s2['q'].new_zeros(3))
assert float(r['heading_sustain'][0])==0.0
s2['yaw'][:]=0.
p.update(s2,s2['q'].new_zeros(3))
r=p.update(s2,s2['q'].new_zeros(3))
assert 0<float(r['heading_sustain'][0])<run[-1],'run must restart from zero after a violation'
# Tight scales actually separate near-miss heading errors from good ones.
p2=MovingProgress(n,device,dt,feedback_mode='sustain');p2.active[:]=True;p2.since[:]=3
p3=MovingProgress(n,device,dt,feedback_mode='pose_feedback');p3.active[:]=True;p3.since[:]=3
def path_reward(prog,heading_error):
    s3=dict(s2);s3['yaw']=torch.full((n,),heading_error,device=device)
    prog.reference[:]=0.;prog.raw_reference[:]=0.;prog.raw_yaw[:]=0.;prog.raw[:]=0.;prog.command[:]=0.
    return float(prog.update(s3,s3['q'].new_zeros(3))['path'][0])
tight=path_reward(p2,.20);loose=path_reward(p3,.20)
assert tight<loose,'the sustain scale must punish a 0.20 rad heading error harder'
print('PASS turn-share mixture and dense heading-window credit')

# Losing rear contact after the lift is charged as an event, not as missing support.
q=MovingProgress(n,device,dt);q.active[:]=True;q.since[:]=3;q.lifted[:]=True
s4=dict(s2)
q.rear_prev[:]=True
s4['rear']=torch.ones(n,device=device,dtype=torch.bool)
m1=q.update(s4,s4['q'].new_zeros(3))
assert float(m1['contact_loss'].sum())==0.,'holding contact must not be charged'
s4['rear']=torch.zeros(n,device=device,dtype=torch.bool)
m2=q.update(s4,s4['q'].new_zeros(3))
assert float(m2['contact_loss'].sum())==n,'a lost contact must be charged once'
m3=q.update(s4,s4['q'].new_zeros(3))
assert float(m3['contact_loss'].sum())==0.,'staying off the ground must not be charged twice'
# Before the lift the same signal is not a loss.
q2=MovingProgress(n,device,dt);q2.active[:]=True;q2.since[:]=3;q2.lifted[:]=False;q2.rear_prev[:]=True
m4=q2.update(s4,s4['q'].new_zeros(3))
assert float(m4['contact_loss'].sum())==0.,'no charge before the lift'
print('PASS rear-contact loss is an event reward, once per loss, lift-gated')

# Window-progress credit ramps with the consecutive all-criteria streak.
w=MovingProgress(n,device,dt);w.active[:]=True;w.since[:]=3;w.lifted[:]=True;w.rear_prev[:]=True
sw=dict(s2);sw['rear']=torch.ones(n,device=device,dtype=torch.bool)
vals=[]
for _ in range(20):
    vals.append(float(w.update(sw,sw['q'].new_zeros(3))['window_progress'][0]))
assert vals[0]<vals[5]<vals[-1] and max(vals)<=1.0001,'window credit must ramp with the streak'
sw['yaw'][:]=0.5
assert float(w.update(sw,sw['q'].new_zeros(3))['window_progress'][0])==0.0,'a broken criterion must reset the streak'
print('PASS window-progress credit follows the all-criteria streak and resets on a miss')

# The support window averages the same measurement over the control interval.
v1=MovingProgress(n,device,dt,support_window=3)
v1.active[:]=True;v1.since[:]=3
sv=dict(s2);sv['rear']=torch.ones(n,device=device,dtype=torch.bool)
for _ in range(3):
    v1.update(sv,sv['q'].new_zeros(3))
assert float(v1.result['support'].float().mean())==1.0
sv['rear']=torch.zeros(n,device=device,dtype=torch.bool)
v1.update(sv,sv['q'].new_zeros(3))
assert float(v1.result['support'].float().mean())>0.5,'one missed sample inside a 3-step window must not drop support'
for _ in range(3):
    v1.update(sv,sv['q'].new_zeros(3))
assert float(v1.result['support'].float().mean())==0.0,'losing contact across the window must drop support'
print('PASS support window tolerates a single missed sample but not sustained loss')
