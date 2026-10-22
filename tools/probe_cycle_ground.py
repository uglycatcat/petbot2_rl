"""Normal four-wheel driving calibration, not a policy success evaluation."""
import argparse,json,pathlib
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
try:
 import torch
 from isaaclab.envs import ManagerBasedRLEnv
 from upright.cycle_task import CycleEnvCfg,ensure,state,tensor
 cfg=CycleEnvCfg();cfg.seed=42;cfg.scene.num_envs=64;cfg.sim.device=a.device;cfg.wheel_mode='common_diff';cfg.wheel_slew=480.;cfg.frame_slew=3.
 cfg.terminations.complete=None;cfg.terminations.failure=None;cfg.terminations.timeout=None
 env=ManagerBasedRLEnv(cfg=cfg);env.reset();p=ensure(env);p.active[:]=True;p.phase[:]=4;p.external_mode[:]=True;p.destination[:]=torch.linspace(-.15,.15,64,device=env.device)
 cmd=torch.zeros(64,2,device=env.device);cmd[:,0]=torch.tensor([0,.05,-.05,0],device=env.device).repeat(16);cmd[:,1]=torch.tensor([0,0,.15,-.15],device=env.device).repeat(16);p.set_command(cmd)
 rows=[]
 for tick in range(100):
  env.step(torch.zeros(64,3,device=env.device));s=state(env)
  if tick<30:continue
  h=tensor(env.scene['contact_forces'].data.net_forces_w_history);ids=env.upright_ids;wheel_ids=ids[2]+ids[3];body_ids=ids[0]+ids[1]
  heights=tensor(env.scene['robot'].data.body_pos_w)[:,body_ids,2]-tensor(env.scene.env_origins)[:,2,None]
  rows.append(dict(four=s['four_fraction'].tolist(),per_wheel_duty=(h[:,:,wheel_ids,2]>.05).float().mean(1).tolist(),per_wheel_mean_force=h[:,:,wheel_ids,2].clamp_min(0).mean(1).tolist(),height=heights.tolist(),body=s['body_history'].tolist(),vz=s['front_vertical_speed'].tolist(),frame=s['q'][:,3:].tolist()))
 pathlib.Path(a.output).write_text(json.dumps(rows));print('Ground calibration samples',len(rows)*64);env.close()
except BaseException:
 import traceback,sys,os
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
