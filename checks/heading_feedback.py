"""CUDA analytic derivative and single-factor reward regression."""
import pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.moving import projected_heading_rate,MovingProgress
assert torch.cuda.is_available()
device='cuda:0';torch.manual_seed(20261022)
f=torch.randn(4096,3,device=device,dtype=torch.float64);f=f/f.norm(dim=-1,keepdim=True)
f=f[f[:,:2].norm(dim=-1)>.3];w=torch.randn_like(f)
speed=w.norm(dim=-1,keepdim=True);axis=w/speed
# Central finite difference under exact world-axis Rodrigues rotation.
def rotate(dt):
 t=speed*dt
 return f*t.cos()+torch.cross(axis,f,dim=-1)*t.sin()+axis*(axis*f).sum(-1,keepdim=True)*(1-t.cos())
eps=1e-6;plus=rotate(eps);minus=rotate(-eps)
delta=torch.atan2(plus[:,1],plus[:,0])-torch.atan2(minus[:,1],minus[:,0])
fd=torch.atan2(delta.sin(),delta.cos())/(2*eps)
actual=projected_heading_rate(f,w)
assert torch.allclose(actual,fd,atol=1e-7,rtol=1e-7)
print('PASS projected heading derivative: max finite-difference error',float((actual-fd).abs().max()))
s=dict(pos=torch.tensor([[0.,0.,.1]],device=device),yaw=torch.tensor([.2],device=device),forward=torch.zeros(1,device=device),vel=torch.tensor([[0.,0.,0.,.1,.2,.2]],device=device),q=torch.zeros(1,5,device=device),g=torch.tensor([[.7,0.,-.714]],device=device),front_z=torch.full((1,),.18,device=device),rear=torch.ones(1,device=device,dtype=torch.bool),other=torch.zeros(1,device=device,dtype=torch.bool),four=torch.zeros(1,device=device,dtype=torch.bool),heading_rate=torch.tensor([.4],device=device))
outputs=[]
for mode in ('pose_feedback_tight','pose_feedback_projected'):
 p=MovingProgress(1,device,.06,feedback_mode=mode);p.active[:]=True;p.since[:]=3
 outputs.append(p.update(s,torch.zeros(3,device=device)))
for k in outputs[0]:
 if k!='tracking':assert torch.equal(outputs[0][k],outputs[1][k]),k
assert not torch.equal(outputs[0]['tracking'],outputs[1]['tracking'])
# At a level pose with equal derivatives the entire reward/criteria state agrees.
s['heading_rate']=s['vel'][:,5]
outputs=[]
for mode in ('pose_feedback_tight','pose_feedback_projected'):
 p=MovingProgress(1,device,.06,feedback_mode=mode);p.active[:]=True;p.since[:]=3
 outputs.append(p.update(s,torch.zeros(3,device=device)))
assert all(torch.equal(outputs[0][k],outputs[1][k]) for k in outputs[0])
print('PASS only tracking changes when derivatives differ; all terms identical when equal')
