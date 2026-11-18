"""Execute a checkpoint or exported raw policy in the same independent PhysX task.
Use --video to save actual execution frames; no scripted robot motion is introduced.
"""
from pathlib import Path
import argparse
import sys
import json
import hashlib
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--policy',type=Path,required=True)
p.add_argument('--scripted',action='store_true',help='Policy is policy_raw.pt TorchScript export')
p.add_argument('--asset-dir',type=Path,default=Path(__file__).parent/'assets')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--video',action='store_true')
p.add_argument('--num-envs',type=int,default=1)
p.add_argument('--seed',type=int,default=9901)
p.add_argument('--step-height',type=float,default=0.07,help='Platform height in metres; use0.05 for the original release')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if not 0<a.step_height<1:p.error('--step-height must be between0 and1 metre')
if a.video:a.enable_cameras=True
app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from step_explore.env import ExploreCfg,ExploreEnv
from step_explore.policy import load_actor
from step_explore.evaluate import evaluate

a.output.mkdir(parents=True,exist_ok=True)
frames=[];observations=[]
writer=None
try:
    cfg=ExploreCfg();cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs;cfg.height=a.step_height
    cfg.sim.device=a.device;cfg.asset_dir=str(a.asset_dir.resolve())
    cfg.viewer.eye=(1.25,-1.3,.8);cfg.viewer.lookat=(.6,0.,.12)
    cfg.viewer.resolution=(960,640)
    class PlaybackEnv(ExploreEnv):
        frame_sink=None
        def _setup_scene(self):
            super()._setup_scene()
            if a.video:
                # Display materials only; preserve all dedicated physics-material bindings.
                from pxr import UsdGeom,UsdShade,Sdf
                for prim in list(self.sim.stage.Traverse()):
                    if prim.IsA(UsdGeom.Mesh) and '/Robot/' in str(prim.GetPath()):
                        mesh=UsdGeom.Mesh(prim)
                        colors=mesh.GetDisplayColorAttr().Get()
                        if not colors: continue
                        name=str(prim.GetPath()).replace('/','_')
                        material=UsdShade.Material.Define(self.sim.stage,'/World/PlaybackMaterials/'+name)
                        shader=UsdShade.Shader.Define(self.sim.stage,str(material.GetPath())+'/Surface')
                        shader.CreateIdAttr('UsdPreviewSurface')
                        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(colors[0])
                        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.65)
                        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
                        mesh.CreateDoubleSidedAttr(True)
        def _get_rewards(self):
            reward=super()._get_rewards()
            if self.frame_sink is not None:
                self.frame_sink(self.render())
            return reward
    env=PlaybackEnv(cfg,render_mode='rgb_array' if a.video else None)
    wrapped=RslRlVecEnvWrapper(env)
    if a.scripted:
        model=torch.jit.load(str(a.policy),map_location=a.device).eval()
        actor=lambda obs:model(obs['policy'])
    else:
        model,_=load_actor(a.policy,a.device)
        actor=model.act_inference
    if a.video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(str(a.output/'execution.mp4'),fps=50,codec='libx264',quality=8,
                                  macro_block_size=16)
        env.frame_sink=writer.append_data
    def policy(obs):
        observations.append(obs['policy'].cpu().numpy().copy())
        return actor(obs)
    result=evaluate(env,wrapped,policy,a.output,a.num_envs)
    np.savez_compressed(a.output/'policy_inputs.npz',observations=np.stack(observations))
    metadata={'policy_sha256':hashlib.sha256(a.policy.read_bytes()).hexdigest(),'scripted':a.scripted,
              'seed':a.seed,'result':result,'video':'execution.mp4' if a.video else None,
              'video_note':'Actual post-physics frames are recorded before reset, including the terminal state.'}
    (a.output/'playback.json').write_text(json.dumps(metadata,indent=2))
    env.close()
except BaseException as exc:
    import traceback
    traceback.print_exc()
    (a.output/'error.json').write_text(json.dumps({'error':repr(exc)}))
    if 'env' in globals():env.close()
    raise
finally:
    if writer is not None:writer.close()
    sys.stdout.flush();sys.stderr.flush();app.close(wait_for_replicator=False,skip_cleanup=True)
