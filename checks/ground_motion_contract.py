"""CUDA analytic checks: pitch must not become wheel-ground steering."""
import pathlib
import sys
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.ground_motion import wheel_heading,lateral_axis

assert torch.cuda.is_available()
device='cuda:0';dtype=torch.float64
def multiply(a,b):
    av,aw=a[...,:3],a[...,3:];bv,bw=b[...,:3],b[...,3:]
    return torch.cat((aw*bv+bw*av+torch.cross(av,bv,dim=-1),aw*bw-(av*bv).sum(-1,keepdim=True)),-1)
def rotation(axis,angle):
    angle=torch.as_tensor(angle,device=device,dtype=dtype).reshape(-1,1)
    axis=torch.as_tensor(axis,device=device,dtype=dtype).expand(len(angle),3)
    return torch.cat((axis*(angle/2).sin(),(angle/2).cos()),-1)
pitch=torch.linspace(-1.4,1.4,65,device=device,dtype=dtype)
q=multiply(multiply(rotation([0,0,1],.7),rotation([1,0,0],.1)),rotation([0,1,0],pitch))
velocity=torch.zeros(65,6,device=device,dtype=dtype)
velocity[:,3:]=lateral_axis(q)*.8
forward,heading,rate=wheel_heading(q,velocity)
assert (heading-heading[0]).abs().max()<1e-12
assert rate.abs().max()<1e-12
velocity[:,3:]=velocity.new_tensor([0.,0.,.2])
assert (wheel_heading(q,velocity)[2]-.2).abs().max()<1e-12
torch.manual_seed(9);omega=torch.randn(65,3,device=device,dtype=dtype)*.2
velocity[:,3:]=omega;eps=1e-6
norm=omega.norm(dim=-1,keepdim=True)
dq=torch.cat((omega/norm*(norm*eps/2).sin(),(norm*eps/2).cos()),-1)
next_heading=wheel_heading(multiply(dq,q),velocity)[1]
derivative=torch.atan2((next_heading-heading).sin(),(next_heading-heading).cos())/eps
assert (derivative-wheel_heading(q,velocity)[2]).abs().max()<1e-6
print('PASS wheel-ground heading invariant to pitch, exact yaw rate, finite-difference derivative')
