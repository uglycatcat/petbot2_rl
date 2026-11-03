"""Check held input across a real landing and zero input after an injected EXIT."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from petbot_s2s.restart import RestartSimulation
from petbot_s2s.restart_core.controller import Phase
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
p.add_argument('--policy',type=Path,default=ROOT/'policies/restart20_s102');args=p.parse_args()
torch.set_num_threads(1)
result={}
sim=RestartSimulation(args.policy);c=sim.controller
c.set_command(.15,0.)
arm=land=False;balance=complete=None;rows=[]
while sim.data.time<22:
    t=float(sim.data.time)
    # Same command value before, during and after the transfer (no new key event).
    c.set_command(.15 if complete is None or t<complete+3 else 0.,0.)
    if t>=3 and not arm:c.toggle();arm=True
    if int(c.control.phase[0])==Phase.BALANCE and balance is None:balance=t
    if balance is not None and t-balance>=6 and not land:c.toggle();land=True
    sim.step();rows.append(sim.record())
    if c.cycles and complete is None:complete=float(sim.data.time)
assert c.cycles==1 and c.interruptions==0, c.events
held=[r for r in rows if complete+1<r['time']<complete+3]
idle=[r for r in rows if r['time']>complete+5]
assert len(held)>30 and len(idle)>30
assert all(abs(r['raw'][0]-.15)<1e-6 for r in held)
assert np.mean([r['motion'][0] for r in held])>.1
assert all(r['raw']==[0.,0.] for r in idle)
assert max(abs(r['motion'][0]) for r in idle)<.005
handoff=[r for r in rows if r['phase']==Phase.HANDOFF and r['phase_time']>0]
assert handoff and all(abs(r['raw'][0]-.15)<1e-6 for r in handoff)
result['held_command']={'cycles':c.cycles,'held_measured_mean':float(np.mean([r['motion'][0] for r in held])),
                        'released_speed_max':max(abs(r['motion'][0]) for r in idle),'unchanged_input_owned_during_transfer':True}
# Release while the hips/ground are still being confirmed, not after success.
sim=RestartSimulation(args.policy);c=sim.controller
arm=land=False;balance=handoff_at=released=None;rows=[]
while sim.data.time<16:
    t=float(sim.data.time)
    release=handoff_at is not None and t>=handoff_at+.12
    c.set_command(0. if release else .15,0.)
    if release and released is None:released=t
    if t>=3 and not arm:c.toggle();arm=True
    if int(c.control.phase[0])==Phase.BALANCE and balance is None:balance=t
    if balance is not None and t-balance>=6 and not land:c.toggle();land=True
    if int(c.control.phase[0])==Phase.HANDOFF and handoff_at is None:handoff_at=t
    sim.step();rows.append(sim.record())
assert c.cycles==1 and c.interruptions==0,c.events
after=[r for r in rows if r['time']>released+.001]
assert after and all(r['raw']==[0.,0.] for r in after)
assert any(r['phase']==Phase.HANDOFF for r in after)
# Records follow physics; use the bundle period for the float32 blend boundary.
blend=c.cfg.handoff_blend_s
assert max(abs(x) for r in after if r['time']>=handoff_at+blend+c.dt-1e-8 for x in r['wheel_target'])<1e-5
assert all(r['wheel_target']==[0.,0.] for r in after if r['time']>=handoff_at+blend+2*c.dt-1e-8)
assert max(abs(r['motion'][0]) for r in after if r['time']>handoff_at+.65)<.005
result['release_during_handoff']={'cycles':c.cycles,'release_at':released,'handoff_at':handoff_at,
                                 'zero_owned_before_success':True,'no_deferred_command':True}
# Exercise the adapter's unexpected-exit branch, explicitly a synthetic event,
# then use real MuJoCo steps to verify initial continuity and zero-speed ownership.
sim=RestartSimulation(args.policy);c=sim.controller
c.set_command(-.15,0.)
for _ in range(100):sim.step()
c.set_command(0.,0.);c.toggle();sim.step()
s=sim.state();s['g'][:]=torch.tensor([[0.,1.,0.]])
previous=c.control.executed.clone()
c.end(s,float(sim.data.time),False,.15)
assert int(c.control.phase[0])==Phase.EXIT and not c.armed
c.set_command(0.,0.);sim.step()
assert torch.equal(previous[:,1:],c.control.executed[:,1:])
for _ in range(100):c.set_command(0.,0.);sim.step()
r=sim.record()
assert r['phase']==Phase.EXIT and not r['armed'] and r['interruptions']==1
assert r['raw']==[0.,0.] and r['wheel_target']==[0.,0.] and abs(r['motion'][0])<.005
result['injected_exit']={'first_wheel_targets_inherited':True,'zero_request_owned':True,'no_auto_rise':True,'final_speed':r['motion'][0]}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
