"""One-step credit preserves the original good and held definitions."""
import pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.moving import MovingProgress
assert torch.cuda.is_available();device='cuda:0'
s=dict(pos=torch.tensor([[0.,0.,.1]],device=device),yaw=torch.zeros(1,device=device),forward=torch.zeros(1,device=device),vel=torch.zeros(1,6,device=device),q=torch.zeros(1,5,device=device),g=torch.tensor([[0.,0.,-1.]],device=device),front_z=torch.full((1,),.18,device=device),rear=torch.ones(1,device=device,dtype=torch.bool),other=torch.zeros(1,device=device,dtype=torch.bool),four=torch.zeros(1,device=device,dtype=torch.bool))
p=MovingProgress(1,device,.06);p.active[:]=True;p.since[:]=3;fixed=torch.zeros(3,device=device)
for _ in range(32):m=p.update(s,fixed)
assert m['held'].item() and m['instant_good'].item()
s['rear'][:]=False;m=p.update(s,fixed)
assert not m['held'].item() and not m['instant_good'].item()
s['rear'][:]=True
for _ in range(p.window-1):
 m=p.update(s,fixed);assert m['instant_good'].item() and not m['held'].item()
m=p.update(s,fixed);assert m['held'].item()
s['other'][:]=True;m=p.update(s,fixed);assert not m['instant_good'].item()
p.reset(torch.tensor([0],device=device),s['pos'],s['yaw']);s['other'][:]=False
m=p.update(s,fixed);assert not m['instant_good'].item() and not m['held'].item()
print('PASS current-step credit, unchanged rolling window, other-contact rejection and reset')
