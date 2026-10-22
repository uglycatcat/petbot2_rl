"""Replay recorded Isaac configurations in unchanged MJCF; no policy or simulation rollout."""
import argparse,json,pathlib,os
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco,numpy as np,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('directory',type=pathlib.Path);p.add_argument('--env',type=int,default=0);a=p.parse_args()
d=np.load(a.directory/'physics_trace.npz');meta=json.loads((a.directory/'replay_meta.json').read_text());root=pathlib.Path(__file__).resolve().parents[3]
model=mujoco.MjModel.from_xml_path(str(root/'robot_description/mjcf/scene.xml'));data=mujoco.MjData(model)
ids=[]
for n in meta['joint_names']:
 if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)<0:n=n.removesuffix('_1')
 ids.append(int(model.joint(n).qposadr[0]))
idx=a.env;active=np.where(d['valid'][:,idx])[0];assert len(active)>0
last=active[-1];start=active[0];snaps=[0,start,min(start+4,last),min(start+8,last),min(start+17,last),last];images=[]
out=a.directory/f'visual_env{idx}';out.mkdir(exist_ok=True)
r=mujoco.Renderer(model,height=360,width=640);cam=mujoco.MjvCamera();cam.distance=.85;cam.azimuth=135;cam.elevation=-15
with imageio.get_writer(out/'recorded_states.mp4',fps=1/meta['control_dt'],macro_block_size=1) as writer:
 for t in range(last+1):
  data.qpos[:3]=d['pos'][t,idx];q=d['quat'][t,idx];data.qpos[3:7]=q[[3,0,1,2]];data.qpos[ids]=d['qfull'][t,idx]
  mujoco.mj_forward(model,data);cam.lookat[:]=data.xpos[model.body('lower_body').id];r.update_scene(data,camera=cam);im=r.render().copy();writer.append_data(im)
  if t in snaps:images.append((t,im))
r.close();fig,axs=plt.subplots(2,3,figsize=(13,6))
for ax,(t,im) in zip(axs.flat,images):ax.imshow(im);ax.set_title(f't={(t+1)*meta["control_dt"]:.2f}s; active={d["since"][t,idx]:.2f}s');ax.axis('off')
fig.suptitle('Recorded Isaac states in unchanged MJCF — NOT policy sim2sim');fig.tight_layout();fig.savefig(out/'frames.png',dpi=130);plt.close(fig)
fig,axs=plt.subplots(3,1,figsize=(11,7),sharex=True);time=(np.arange(last+1)+1)*meta['control_dt']
axs[0].plot(time,d['front_z'][:last+1,idx],label='front min height (m)');axs[0].plot(time,d['rear_z'][:last+1,idx],label='rear max center height (m)')
axs[1].plot(time,d['force'][:last+1,idx].min(axis=-1).mean(axis=-1),label='min rear force averaged over physical steps (N)');axs[1].plot(time,d['force'][:last+1,idx,-1].min(axis=-1),label='min rear force at control sample (N)',alpha=.65)
axs[2].plot(time,d['support'][:last+1,idx],label='raw support');axs[2].plot(time,d['held'][:last+1,idx],label='rolling held')
for ax in axs:ax.legend();ax.grid(alpha=.3)
axs[-1].set_xlabel('Simulation time (s)');fig.tight_layout();fig.savefig(out/'signals.png',dpi=130);plt.close(fig)
print(out)
