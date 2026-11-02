"""ONNX CPU policy with real MuJoCo IMU/encoder feedback."""
import hashlib,json
from pathlib import Path
import mujoco
import numpy as np
import onnxruntime as ort
from .control import Controller
from .estimator import Estimator
from .monitor import Monitor,WHEELS,terrain_clearance
from .scene import load_scene

PROJECT=Path(__file__).resolve().parents[1]


class Preview:
    def __init__(self,policy=None,model_dir=None,continue_on_failure=True):
        if policy is None:
            selection=json.loads((PROJECT/'policies/selection.json').read_text());policy=PROJECT/'policies'/selection['default']
        policy=Path(policy);self.metadata=json.loads((policy/'policy.json').read_text());m=self.metadata
        if hashlib.sha256((policy/'policy.onnx').read_bytes()).hexdigest()!=m['onnx_sha256']:raise ValueError('ONNX hash mismatch')
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(policy/'policy.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        if self.session.get_inputs()[0].shape[-1]!=26:raise ValueError('Expected 26-dimensional Step observation')
        self.model,self.edge=load_scene(model_dir or PROJECT.parents[1]/'robot_description',m);self.data=mujoco.MjData(self.model)
        self.dt=float(self.model.opt.timestep);self.decimation=round(m['policy_dt']/self.dt)
        if abs(self.dt-m['physics_dt'])>1e-12 or abs(self.decimation*self.dt-m['policy_dt'])>1e-12:raise ValueError('Control timing mismatch')
        self.names=m['joint_names'];self.jids=[self.model.joint(n).id for n in self.names]
        self.qids=self.model.jnt_qposadr[self.jids];self.vids=self.model.jnt_dofadr[self.jids]
        self.hips=[self.names.index(x) for x in ['left_frame','right_frame']];self.wheels=[self.names.index(x) for x in ['left_rear_wheel','right_rear_wheel']]
        self.root=self.model.body('base_link').id;self.lower=self.model.body('lower_body').id
        self.gids=[self.model.geom(n).id for n in WHEELS];self.bids=[self.model.body(n).id for n in WHEELS]
        self.ground=self.model.geom('floor').id;self.platform=self.model.geom('step_platform').id
        self.vertices=[np.asarray(m['asset']['wheel_vertices'][n]) for n in WHEELS]
        self.edges=[np.asarray(m['asset']['rear_edges'][n],int) for n in WHEELS[2:]]
        self.jac=np.zeros((3,self.model.nv));self.bodyvel=np.zeros(6)
        self.estimator=Estimator(m['asset']['source_bodies'],self.names);self.controller=Controller(m['training']);self.monitor=Monitor(m,self.edge)
        self.continue_on_failure=continue_on_failure;self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model,self.data);mujoco.mj_forward(self.model,self.data)
        self.estimator.reset();self.controller.reset();self.monitor.reset()
        self.packet=np.array([.02,.30,np.pi/2],np.float32);self.packet_count=0;self.policy_calls=0;self.state='IDLE'
        self.substeps=0;self.last_observation=np.zeros(26,np.float32);self.failure_snapshot=None
        self.origin=np.asarray(self.metadata['asset']['initial_com']);self._geometry()

    def start(self):
        if self.state!='IDLE':return False
        self.packet_count=1;self.state='RUNNING';return True

    def key(self,key):
        if key==32:return self.start()
        if key in (ord('R'),ord('r')):self.reset();return True
        return False

    def joints(self):return self.data.qpos[self.qids].astype(np.float32),self.data.qvel[self.vids].astype(np.float32)

    def _servo(self):
        h,w=self.controller.advance(self.dt);t=self.metadata['training']
        for name in ['head','left_ear','right_ear','left_frame','right_frame']:
            j=self.model.joint(name);i=self.model.actuator(name).id
            group='hip' if name.endswith('frame') else 'ear' if 'ear' in name else 'head'
            target=h if group=='hip' else 0.
            torque=t[group+'_kp']*(target-self.data.qpos[j.qposadr[0]])-t[group+'_kd']*self.data.qvel[j.dofadr[0]]
            self.data.ctrl[i]=np.clip(torque,*self.model.actuator_ctrlrange[i])
        for name in ['left_rear_wheel','right_rear_wheel']:self.data.ctrl[self.model.actuator(name).id]=w

    def _geometry(self):
        points=[v@self.data.xmat[b].reshape(3,3).T+self.data.xpos[b] for v,b in zip(self.vertices,self.bids)]
        self.minimum=np.array([x.min(0)-.0001 for x in points]);self.maximum=np.array([x.max(0)+.0001 for x in points])
        t=self.metadata['task']['terrain']
        self.clearance=np.array([terrain_clearance(x,e,self.edge,t['height_m'],t['platform_length_m'],t['platform_width_m']/2) for x,e in zip(points[2:],self.edges)])
        self.com=self.data.subtree_com[self.root].copy()-self.origin
        mujoco.mj_jacSubtreeCom(self.model,self.data,self.jac,self.root);self.velocity=self.jac@self.data.qvel

    def _contacts(self):
        force=np.zeros((4,3));platform=np.zeros((4,3));cx=np.full(4,np.nan);body_forces={};contact_force=np.zeros(6)
        for j in range(self.data.ncon):
            c=self.data.contact[j];g1,g2=int(c.geom1),int(c.geom2)
            mujoco.mj_contactForce(self.model,self.data,j,contact_force);f=c.frame.reshape(3,3).T@contact_force[:3]
            for g,sign,other in [(g1,-1,g2),(g2,1,g1)]:
                if g in self.gids:
                    i=self.gids.index(g);force[i]+=sign*f
                    if other==self.platform:
                        platform[i]+=sign*f;cx[i]=c.pos[0] if np.isnan(cx[i]) else min(cx[i],c.pos[0])
                elif g not in (self.ground,self.platform):body_forces[g]=body_forces.get(g,np.zeros(3))+sign*f
        body_hit=any(np.linalg.norm(v)>.5 for g,v in body_forces.items() if self.model.geom(g).name in ['head','left_frame','right_frame','lower_body'])
        self.force=force;self.platform_force=platform;self.contact_x=cx
        return body_hit

    def step(self):
        if self.state!='RUNNING':return self.snapshot()
        q,v=self.joints();gyro=self.data.sensor('imu_gyro').data.astype(np.float32).copy()
        self.last_observation=self.estimator.observation(self.packet,q,v,gyro,self.controller.executed)
        action=self.session.run(['action'],{'observation':self.last_observation[None]})[0][0]
        self.controller.request(action);self.policy_calls+=1
        for _ in range(self.decimation):
            self._servo();mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data)
            q,v=self.joints();gyro=self.data.sensor('imu_gyro').data.astype(np.float32).copy()
            self.estimator.update(self.data.sensor('imu_accel').data,gyro,q,self.dt,v)
            self.substeps+=1;self._geometry();body_hit=self._contacts()
            self.monitor.capture(self.minimum,self.maximum,self.force,self.platform_force,self.contact_x,self.clearance,
                self.com,self.data.xmat[self.lower].reshape(3,3)[2,2],body_hit,v[self.wheels],self.dt)
            if self.monitor.failure and self.failure_snapshot is None:self.failure_snapshot=self.snapshot()
        rotation=self.data.xmat[self.lower].reshape(3,3);yaw=np.arctan2(rotation[1,0],rotation[0,0])
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.lower,self.bodyvel,0)
        self.monitor.control_step(self.minimum,self.maximum,self.com,self.velocity,yaw,self.bodyvel[2],q[self.hips],self.metadata['policy_dt'])
        if self.monitor.failure and not self.continue_on_failure:self.state='FAILED'
        elif self.monitor.completed:self.state='COMPLETE'
        elif self.data.time+1e-8>=self.metadata['task']['acceptance']['timeout_s']:self.state='TIMEOUT'
        return self.snapshot()

    def snapshot(self):
        q,v=self.joints();mo=self.monitor
        return dict(state=self.state,time=float(self.data.time),phase=mo.phase,failure=mo.failure,completed=mo.completed,
            policy_calls=self.policy_calls,packet_count=self.packet_count,com=self.com.tolist(),velocity=self.velocity.tolist(),
            estimated_com=self.estimator.com_position.tolist(),estimated_velocity=self.estimator.com_velocity.tolist(),
            action=self.controller.action.tolist(),executed=self.controller.executed.tolist(),hip_target=float(self.controller.hip),
            wheel_target=float(self.controller.wheel),joint_position=q.tolist(),joint_velocity=v.tolist(),
            front_lifted=mo.lifted.tolist(),front_cleared=mo.cleared.tolist(),front_landed=mo.landed.tolist(),
            supported_top=mo.supported.tolist(),rear_clearance=self.clearance.tolist(),rear_air=mo.rear_air.tolist(),
            ever_fourtop=mo.ever_fourtop,ever_home=mo.ever_home,ever_drive=mo.ever_drive,
            max_actual_wheel_speed=mo.peak_wheel_speed,peak_contact_force=mo.peak_contact)
