"""One-environment renderer check with fixed motor commands. No PPO or weights."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.video=True;a.enable_cameras=True
    a.output.mkdir(parents=True,exist_ok=False);app=AppLauncher(a).app;env=None;video=None
    try:
        import torch
        from step_demo.config import PROJECT_ROOT,load_config
        from step_demo.runtime import load_training
        from step_demo.isaac_env import StepEnv
        from step_demo.video import PolicyVideo
        t=load_training(PROJECT_ROOT/'configs/engineering.json');t['num_envs']=1
        env=StepEnv(load_config(model_dir=a.model_dir),t,a.asset_dir,a.device,render_mode='rgb_array')
        env.trace_ids=[0]
        with torch.inference_mode():
            env.reset();video=PolicyVideo(env,a.output/'render_check.mp4','RENDER CHECK - FIXED COMMANDS - NO TRAINING')
            video.begin()
            for k in range(env.max_episode_length+1):
                env.step(torch.tensor([[0.,-.64]],device=a.device));video.step(k+1)
                if env.terminal_records:break
            if not env.terminal_records:raise RuntimeError('Probe did not produce a terminal frame')
            result=video.finish(env.terminal_records[0])
        result.update(status='passed',probe_only=True,ppo_used=False,weights_loaded=False,training_started=False)
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        if video is not None:video.close()
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
