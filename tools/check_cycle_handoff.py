"""PhysX handoff plumbing check from an already grounded robot, not learned landing."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
try:
 import torch
 from isaaclab.envs import ManagerBasedRLEnv
 from upright.cycle_task import CycleEnvCfg,SPEC,ensure
 cfg=CycleEnvCfg();cfg.scene.num_envs=16;cfg.sim.device=a.device;cfg.wheel_mode='common_diff';cfg.wheel_slew=120.;cfg.frame_slew=3.
 env=ManagerBasedRLEnv(cfg=cfg);env.reset();progress=ensure(env)
 progress.active[:]=True;progress.phase[:]=3;progress.mode[:]=0;progress.destination[:]=0.;progress.external_mode[:]=True
 progress.set_command([0.,0.]);lo,hi=SPEC['joints']['left_frame']['range'];action=torch.zeros(16,3,device=env.device);action[:,0]=-(lo+hi)/(hi-lo)
 seen=torch.zeros(16,device=env.device,dtype=torch.bool);finished=seen.clone()
 for _ in range(50):
  obs,r,term,timeout,extras=env.step(action)
  s,m,*_=env.moving_snapshot
  seen |= m['entered_handoff'];finished |= m['completed']
  assert not (term&timeout).any(),'Failure/timeout overlap'
  assert torch.isfinite(obs['policy']).all() and torch.isfinite(r).all()
  if finished.all():break
 assert seen.all() and finished.all(),(int(seen.sum()),int(finished.sum()))
 print('PASS PhysX grounded landing admission -> stable handoff -> ordinary control observation,16/16; NOT learned landing')
 env.close()
except BaseException:
 import traceback,sys,os
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
