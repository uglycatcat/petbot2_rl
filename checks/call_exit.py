"""MuJoCo fault-injection check of interruption and explicit re-entry semantics."""
import argparse,json,sys
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from petbot_s2s.simulation import Simulation
p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
results={}
for fault in ('front_return','overturned'):
 sim=Simulation(a.policy);c=sim.controller;progress=c.progress
 assert c.full_range
 for _ in range(30):sim.step()
 c.armed=True;progress.active[:]=True;progress.phase[:]=2;progress.phase_time[:]=2.;progress.lift_seen[:]=True;progress.request_mode(1,c.frame)
 if fault=='overturned':
  sim.data.qpos[3:7]=[0.,1.,0.,0.];sim.data.qvel[:]=0.;mujoco.mj_forward(sim.model,sim.data)
 for _ in range(5):sim.step()
 assert c.interruptions==1 and c.cycles==0 and not c.failure_code and not c.armed
 assert int(progress.phase[0])==0 and int(progress.requested_mode[0])==0
 if fault=='overturned':assert not c.events[-1]['ordinary_pose']
 for _ in range(15):sim.step()
 assert c.interruptions==1 and not c.armed and not progress.active.any(), 'Residual mode retriggered upright'
 before=sim.data.qvel.copy();targets=c.action.target.clone();c.toggle()
 assert c.armed and int(progress.requested_mode[0])==1
 assert np.array_equal(sim.data.qvel,before) and (c.action.target==targets).all()
 results[fault]=dict(interruptions=c.interruptions,last_exit_code=c.last_exit_code,ordinary_continued=True,new_request_required=True,physical_state_preserved=True)
result=dict(passed=True,method='MuJoCo state fault injection, not learned recovery quality',cases=results)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
