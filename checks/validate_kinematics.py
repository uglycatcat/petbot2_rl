"""Compare encoder/body-gravity wheel geometry with independent MuJoCo FK."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from petbot_s2s.simulation import Simulation
from petbot_s2s.core.kinematics import wheel_geometry

sim=Simulation(ROOT/'policies/cycle18_long_r42')
m,d=sim.model,sim.data
rng=np.random.default_rng(20261027);max_error=0.
for i in range(200):
    q=rng.normal(size=4);q/=np.linalg.norm(q)
    d.qpos[3:7]=q
    frame=rng.uniform(-1.,1.2,size=2)
    d.qpos[sim.qpos_ids[3:]]=frame
    mujoco.mj_forward(m,d)
    rotation=d.xmat[sim.root].reshape(3,3)
    g=torch.tensor(-rotation[2,:],dtype=torch.float64)[None,:]
    gap,skew,_=wheel_geometry(g,torch.tensor(frame)[None,:])
    expected=d.xpos[sim.front,2]-d.xpos[sim.rear,2]-.015
    expected_skew=abs(d.xpos[sim.rear[0],2]-d.xpos[sim.rear[1],2])
    max_error=max(max_error,float(np.max(np.abs(gap.numpy()[0]-expected))),abs(float(skew)-expected_skew))
assert max_error<1e-12,max_error
print(json.dumps(dict(passed=True,poses=200,max_error_m=max_error,
    sensors=[m.sensor(i).name for i in range(m.nsensor)],
    limitation='Flat-ground rear support is a task assumption; geometry does not measure contact force.'),indent=2))
