"""CUDA: physical landing admits ordinary control, alignment still gates completion."""
import json
import pathlib
import sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from upright.cycle import CycleProgress

assert torch.cuda.is_available()
d='cuda:0'; n=6; dt=.06
z=lambda *shape:torch.zeros(*shape,device=d)
s=dict(pos=z(n,3),yaw=z(n),forward=z(n),vel=z(n,6),q=z(n,5),
       g=z(n,3),front_z=z(n)+.045,rear=torch.ones(n,device=d,dtype=torch.bool),
       other=torch.zeros(n,device=d,dtype=torch.bool),four=torch.ones(n,device=d,dtype=torch.bool),
       rear_fraction=z(n)+1,four_fraction=z(n)+1,
       other_history=torch.zeros(n,device=d,dtype=torch.bool),
       body_history=torch.zeros(n,device=d,dtype=torch.bool),front_vertical_speed=z(n),
       front_force_peak=z(n),front_contact=torch.ones(n,device=d,dtype=torch.bool),heading_rate=z(n))
s['pos'][:,2]=.1;s['g'][:,2]=-1.
s['four_fraction'][1]=.5;s['body_history'][2]=True;s['g'][3,2]=-.7
s['front_vertical_speed'][4]=-.3;s['vel'][5,3]=1.5

def progress(mode):
    p=CycleProgress(n,d,dt,handoff_alignment=mode)
    p.reset(torch.arange(n,device=d),s['pos'],s['yaw'])
    p.active[:]=True;p.phase[:]=3;p.external[:]=True;p.external_mode[:]=True
    p.raw[:]=0;p.command[:]=0;p.destination[:]=.3;p.last_g.copy_(s['g'])
    return p

before=progress('before');after=progress('after')
for step in range(6):
    before.update(s,z(3));m=after.update(s,z(3))
    if step<5:assert (after.phase==3).all(), 'Contact dwell must not be shortened'
assert (before.phase==3).all()
assert after.phase.tolist()==[4,3,3,3,3,3], 'Bad contact/posture/motion must not hand over'
for _ in range(20):m=after.update(s,z(3))
assert not m['completed'].any(), 'Unaligned frame cannot count as completed'
s['q'][0,3:]=.3
for step in range(17):
    m=after.update(s,z(3))
    if step<16:assert not m['completed'].any(), 'Completion requires a full aligned hold'
assert m['completed'].tolist()==[True,False,False,False,False,False]
s['g'][0,2]=.3;m=after.update(s,z(3))
assert m['failure'][0] and not m['completed'][0], 'Failure priority must remain'
after.reset(torch.arange(n,device=d),s['pos'],s['yaw'])
assert not after.handoff.any() and not after.ground_run.any() and not after.handoff_run.any()
print(json.dumps(dict(passed=True,physical_landing_dwell=True,rejects_unstable_contact=True,
                     completion_requires_alignment=True,aligned_hold_seconds=1.,failure_priority=True,reset=True)))
