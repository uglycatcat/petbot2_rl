"""Read-only CUDA probes of deployed task semantics; not a physics acceptance test."""
import json,pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.moving import MovingProgress,slew
assert torch.cuda.is_available()
d='cuda:0';dt=.06
s=dict(pos=torch.tensor([[0.,0.,.1]],device=d),yaw=torch.zeros(1,device=d),forward=torch.zeros(1,device=d),vel=torch.zeros(1,6,device=d),q=torch.zeros(1,5,device=d),g=torch.tensor([[-.8,0.,-.6]],device=d),front_z=torch.full((1,),.18,device=d),rear=torch.ones(1,device=d,dtype=torch.bool),other=torch.zeros(1,device=d,dtype=torch.bool),four=torch.zeros(1,device=d,dtype=torch.bool))
p=MovingProgress(1,d,dt,feedback_mode='pose_feedback_tight');p.active[:]=True;p.external[:]=True;p.raw[:]=torch.tensor([.01,.01],device=d);p.command.copy_(p.raw);p.last_g.copy_(s['g']);fixed=torch.zeros(3,device=d);tail=[]
for _ in range(201):
 p.before_step();m=p.update(s,fixed)
 if p.since.item()>=10:tail.append(m['held'].item())
stationary=dict(held_all_final_2s=all(tail),distance=m['raw_distance'].item(),heading_error=m['heading_error'].item(),requested=[.01,.01])
s['g'][:,2]=.3;m=p.update(s,fixed)
overlap=dict(failure=m['failure'].item(),timeout=(p.active & (p.since>=12)).item())
wheel_jump=(slew(torch.tensor([-25.],device=d),torch.tensor([25.],device=d),900*dt)+25).item()
assert stationary['held_all_final_2s'] and overlap['failure'] and overlap['timeout'] and wheel_jump==50
print(json.dumps(dict(device=d,synthetic_criterion_only=True,stationary=stationary,terminal_overlap=overlap,allowed_wheel_jump_rad_s=wheel_jump,window_steps=p.window,window_sample_span_s=(p.window-1)*dt),indent=2))
