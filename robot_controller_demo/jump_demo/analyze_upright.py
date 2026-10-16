"""Run the original JumpController without GUI; log physics and render its lift/hold.

From the repository root, with the anna-owned MuJoCo environment:
MUJOCO_GL=egl artifacts/mujoco_env/bin/python robot_controller_demo/jump_demo/analyze_upright.py
No model edits, controller modifications, or training data generation.
"""
import argparse, csv, hashlib, json
from pathlib import Path
import mujoco
import numpy as np
from control import JumpController, nose_up

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/jump_demo_analysis');p.add_argument('--seconds',type=float,default=12.);p.add_argument('--render',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
model=mujoco.MjModel.from_xml_path(str(ROOT/'robot_description/mjcf/scene.xml'))
model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
data=mujoco.MjData(model);controller=JumpController(model);mujoco.mj_forward(model,data)
start=data.qpos[:2].copy();dt=model.opt.timestep
front=[model.body(n).id for n in ('left_front_wheel','right_front_wheel')];rear=[model.body(n).id for n in ('left_rear_wheel','right_rear_wheel')]
root=model.body('base_link').id
rows=[];events=[];last=None;snapshots=[];times=[0,1.25,1.8,1.98,2.16,2.34,2.52,2.7,3.,4.,6.,12.]
renderer=None;writer=None
if a.render:
 import imageio.v2 as imageio
 renderer=mujoco.Renderer(model,height=360,width=640)
 cam=mujoco.MjvCamera();cam.distance=1.1;cam.azimuth=105;cam.elevation=-12
 writer=imageio.get_writer(a.output/'jump_demo.mp4',fps=25)

def sample():
 forces={i:0. for i in front+rear};other=0.
 for c in range(data.ncon):
  contact=data.contact[c];g1,g2=contact.geom1,contact.geom2
  if model.geom_type[g1]!=mujoco.mjtGeom.mjGEOM_PLANE and model.geom_type[g2]!=mujoco.mjtGeom.mjGEOM_PLANE:continue
  geom=g2 if model.geom_type[g1]==mujoco.mjtGeom.mjGEOM_PLANE else g1
  body=model.geom_bodyid[geom];f=np.zeros(6);mujoco.mj_contactForce(model,data,c,f)
  if body in forces:forces[body]+=max(0.,f[0])
  else:other+=max(0.,f[0])
 fz,rz=controller.wheel_height(data)
 row=dict(linear_speed=float(np.linalg.norm(data.qvel[:3])), angular_speed=float(np.linalg.norm(data.qvel[3:6])), position_joint_speed=max(abs(float(data.qvel[controller.dadr[n]])) for n in ('head','left_ear','right_ear','left_frame','right_frame')), t=float(data.time),phase=controller.phase,x=float(data.qpos[0]-start[0]),y=float(data.qpos[1]-start[1]),vx=float(data.qvel[0]),pitch=nose_up(data.qpos),pitch_rate=controller._pitch_rate,front_z=fz,rear_z=rz,front_force=sum(forces[i] for i in front),rear_left_force=forces[rear[0]],rear_right_force=forces[rear[1]],other_force=other,com_x=float(data.subtree_com[root,0]),rear_x=float(np.mean(data.xpos[rear,0])),ready=controller.jump_ready)
 for n in ('left_frame','right_frame','left_rear_wheel','right_rear_wheel','head','left_ear','right_ear'):
  row[n+'_q']=float(data.qpos[controller.qadr[n]]);row[n+'_qd']=float(data.qvel[controller.dadr[n]]);row[n+'_ctrl']=float(data.ctrl[controller.act[n]])
 return row
for i in range(round(a.seconds/dt)+1):
 row=sample();rows.append(row)
 if row['phase']!=last:events.append(row.copy());last=row['phase']
 if renderer and (i%20==0 or (times and data.time+dt/2>=times[0])):
  cam.lookat[:]=data.xpos[model.body('lower_body').id];renderer.update_scene(data,camera=cam);img=renderer.render()
  if i%20==0:writer.append_data(img)
  if times and data.time+dt/2>=times[0]:snapshots.append((float(data.time),row['phase'],img.copy()));times.pop(0)
 if i<round(a.seconds/dt):controller.step(data,dt);mujoco.mj_step(model,data)
if writer:writer.close();renderer.close()
with (a.output/'trace.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(6,1,figsize=(11,13),sharex=True)
for ax,keys in zip(axs,[['x','y'],['pitch','left_frame_q'],['left_rear_wheel_ctrl','left_rear_wheel_qd'],['vx','pitch_rate'],['front_z','rear_z'],['front_force','rear_left_force','rear_right_force','other_force']]):
 for k in keys:ax.plot([r['t'] for r in rows],[r[k] for r in rows],label=k)
 for e in events[1:]:ax.axvline(e['t'],color='gray',alpha=.4)
 ax.legend(loc='upper right');ax.grid(alpha=.2)
axs[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(a.output/'trace.png');plt.close(fig)
if snapshots:
 fig,axs=plt.subplots(4,3,figsize=(15,11))
 for ax,(t,phase,img) in zip(axs.flat,snapshots):ax.imshow(img);ax.set_title(f'{t:.2f}s {phase}');ax.axis('off')
 fig.tight_layout();fig.savefig(a.output/'frames.png');plt.close(fig)
tail=[r for r in rows if r['t']>=max(0.,a.seconds-1.5)]
support=lambda r:r['front_z']>.12 and r['rear_left_force']>.05 and r['rear_right_force']>.05 and r['front_force']<.1 and r['other_force']<.1
run=longest=0
for r in rows:run=run+dt if support(r) else 0;longest=max(longest,run)
stable=lambda r:support(r) and r['linear_speed']<.15 and r['angular_speed']<.4 and r['position_joint_speed']<.5
def longest_time(condition):
 run=longest=0.
 for row in rows:
  run=run+dt if condition(row) else 0.;longest=max(longest,run)
 return longest
result=dict(longest_calm_support_s=longest_time(stable), longest_calm_support_at_origin_s=longest_time(lambda r:stable(r) and np.hypot(r['x'],r['y'])<.05), mujoco=mujoco.__version__,seconds=a.seconds,controller_sha256=hashlib.sha256(Path(__file__).with_name('control.py').read_bytes()).hexdigest(),events=events,longest_rear_only_support_s=longest,first_ready_s=next((r['t'] for r in rows if r['ready']),None),final=rows[-1],max_abs_x=max(abs(r['x']) for r in rows),tail={k:[min(r[k] for r in tail),max(r[k] for r in tail)] for k in ['x','vx','pitch','left_frame_q','front_z','rear_z','other_force']})
(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['events','final']},indent=2))
