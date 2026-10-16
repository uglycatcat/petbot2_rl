"""Visualize recorded Isaac state in the unchanged MJCF; this is NOT sim2sim validation."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
out=a.output or a.directory.parent/(a.directory.name+'_visuals');out.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2]
meta=json.loads((a.directory/'evaluation.json').read_text());trace=json.loads((a.directory/'trace.json').read_text())
model=mujoco.MjModel.from_xml_path(str(root/'robot_description/mjcf/scene.xml'));data=mujoco.MjData(model)
renderer=mujoco.Renderer(model,height=368,width=640);cam=mujoco.MjvCamera();cam.distance=1.;cam.azimuth=105;cam.elevation=-12
joint_ids=[]
for name in meta['joint_names']:
 if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)<0:name=name.removesuffix('_1')
 joint=model.joint(name);joint_ids.append(joint.qposadr[0])
active=[r for r in trace if r['active'][0]];snap_indices=set(np.linspace(0,len(active)-1,12,dtype=int));images=[]
with imageio.get_writer(out/'trajectory.mp4',fps=25) as writer:
 for i,r in enumerate(active):
  if i%2 and i not in snap_indices:continue
  data.qpos[:3]=r['pos'][0];xyzw=r['quat'][0];data.qpos[3:7]=[xyzw[3],*xyzw[:3]]
  data.qpos[joint_ids]=r['qfull'][0];mujoco.mj_forward(model,data)
  cam.lookat[:]=data.xpos[model.body('lower_body').id];renderer.update_scene(data,camera=cam);im=renderer.render()
  if i%2==0:writer.append_data(im)
  if i in snap_indices:images.append((r['t'],r['phase'][0],im.copy()))
renderer.close();fig,axs=plt.subplots(4,3,figsize=(15,11))
for ax,(t,phase,im) in zip(axs.flat,images):ax.imshow(im);ax.set_title(f'{t:.2f}s phase {phase}');ax.axis('off')
fig.suptitle('Recorded Isaac states rendered in MJCF (not a MuJoCo policy rollout)');fig.tight_layout();fig.savefig(out/'frames.png');plt.close(fig)
t=[r['t'] for r in active];fig,axs=plt.subplots(4,1,figsize=(10,9),sharex=True)
for i,keys in enumerate([['distance','front_z'],['phase','held'],['action'],['vel']]):
 for key in keys:
  v=np.array([r[key][0] for r in active]);axs[i].plot(t,v,label=key)
 axs[i].legend();axs[i].grid(alpha=.2)
axs[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(out/'trajectory.png');plt.close(fig)
print('Rendered',len(active),'recorded frames')
