"""GPU physics integration: contact direction, history freshness and reset eligibility."""
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
import argparse
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
try:
 import torch
 from isaaclab.envs import ManagerBasedRLEnv
 from upright.moving_task import MovingEnvCfg,ensure,reward,tensor
 from upright.moving import physical_contact_deficit
 h=torch.zeros(4,30,2,3,device=a.device);h[0,:,:,2]=1;h[1,:15,:,2]=1;h[3,:,:,2]=1
 enabled=torch.tensor([True,True,True,False],device=a.device)
 loss=physical_contact_deficit(h,[0,1],enabled)
 assert torch.allclose(loss,torch.tensor([0.,.5,1.,0.],device=a.device))
 cfg=MovingEnvCfg();cfg.seed=42;cfg.sim.device=a.device;cfg.scene.num_envs=64
 cfg.scene.contact_forces.history_length=cfg.decimation;cfg.rewards.physical_contact.weight=-4.
 env=ManagerBasedRLEnv(cfg);env.reset();p=ensure(env)
 for _ in range(35):
  env.step(torch.zeros(64,3,device=a.device))
  sensor=env.scene['contact_forces'];history=tensor(sensor.data.net_forces_w_history)
  assert history.shape==(64,30,len(sensor.body_names),3)
  assert torch.equal(history[:,0],tensor(sensor.data.net_forces_w)), 'History is stale'
  value=reward(env,'physical_contact')
  assert torch.isfinite(value).all() and (value>=0).all() and (value<=1).all()
  assert (value*cfg.rewards.physical_contact.weight*env.step_dt<=0).all()
 # Explicitly prime lifted state, then reset all environments; no episode leakage.
 p.active[:]=True;p.lifted[:]=True
 env.reset()
 assert not p.lifted.any() and not p.active.any()
 assert not reward(env,'physical_contact').any()
 print('PASS CUDA contact deficit 0/.5/1, negative integrated reward, latest history and reset gate')
 env.close()
except BaseException:
 import traceback,os,sys
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
