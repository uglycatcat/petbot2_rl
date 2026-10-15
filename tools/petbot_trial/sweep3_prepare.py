"""Prepare the full training library and open-loop curriculum labels without PPO updates."""
import argparse
import json
import os
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();app=AppLauncher(a).app

def main():
    import gymnasium as gym
    import agile.rl_env.tasks.stand_up.petbot
    from isaaclab_tasks.utils import load_cfg_from_registry
    from agile.petbot.pre_learn import pre_learn
    task='PetBot-Recovery-Sweep3-C-v0'
    cfg=load_cfg_from_registry(task,'env_cfg_entry_point');cfg.scene.num_envs=2048;cfg.sim.device='cuda:0'
    agent=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point');agent.seed=42;agent.device='cuda:0'
    env=gym.make(task,cfg=cfg);pre_learn(env.unwrapped,task,agent)
    Path(a.output).write_text(json.dumps(env.unwrapped.petbot_provenance,indent=2)+'\n')
    env.close()

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    app.close()
