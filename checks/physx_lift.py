"""Diagnostic position/velocity excitation; never training resets or demonstration data."""
import argparse,json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab.envs import ManagerBasedRLEnv
from upright.task import UprightEnvCfg,SPEC,state
try:
 cfg=UprightEnvCfg();cfg.scene.num_envs=64;cfg.sim.device=a.device
 env=ManagerBasedRLEnv(cfg=cfg);env.reset();env.upright_record=True
 n=env.num_envs;idx=torch.arange(n,device=env.device);peak=torch.zeros(n,device=env.device);seen=peak.bool();active=torch.ones_like(seen)
 # Variations are a physical interface probe, not an action reference installed in PPO.
 prep_duration=.3+.1*(idx%4);prep_frame=.6+.2*((idx//4)%4);reverse=-10.-5.*((idx//16)%4)
 lo,hi=SPEC['joints']['left_frame']['range'];kv=SPEC['actuators']['left_rear_wheel']['kv'];limit=SPEC['actuators']['left_rear_wheel']['control_range'][1]
 for step in range(300):
  t=step*env.step_dt;s=state(env)
  pitch=torch.asin((-s['g'][:,0]).clamp(-1,1));pitch_rate=-s['vel'][:,4]
  torque=-(1.15*(-2.1775447*s['vel'][:,0]+41.85550149*(pitch-.614)+2.5513757*pitch_rate)).clamp(-.55,.55)
  balance_speed=s['wheel'].mean(-1)+torque*.5/kv
  speed=torch.where(t<prep_duration,reverse,torch.where(t<prep_duration+.12,torch.full_like(reverse,25.),balance_speed))
  frame=torch.where(t<prep_duration,prep_frame,torch.full_like(prep_frame,-.4))
  action=torch.stack(((2*frame-lo-hi)/(hi-lo),speed.clamp(-limit,limit)/limit),-1)
  _,_,term,timeout,_=env.step(action)
  s,m=env.upright_snapshot
  peak=torch.maximum(peak,torch.where(active,s['front_z'],0.));seen |= active & (m['phase']==1)
  active &= ~(term|timeout)
 result={'position_velocity_interface':True,'max_front_height':float(peak.max()),'phase2_cases':int(seen.sum()),'cases':64,'peaks':peak.tolist(),'phase2':seen.tolist()}
 Path(a.output).write_text(json.dumps(result,indent=2));print(result)
 env.close()
finally:app.close()
