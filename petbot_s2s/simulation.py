"""Read-only MJCF loading, state translation and external position servos."""
from collections import deque
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch

from .controller import Controller
from .core.ground_motion import wheel_heading


def find_model():
    for parent in Path(__file__).resolve().parents:
        candidate = parent/'robot_description'
        if (candidate/'mjcf/scene.xml').is_file():
            return candidate
    raise FileNotFoundError('Pass --model-dir or keep this checkout in the PetBot workspace')


def fingerprint(directory):
    digest = hashlib.sha256()
    for path in sorted([*directory.glob('mjcf/*.xml'),*directory.glob('meshes/*')]):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def batch(value, dtype=torch.float32):
    return torch.as_tensor(np.array(value,copy=True),dtype=dtype).unsqueeze(0)


class Simulation:
    def __init__(self, bundle, model_dir=None, frame=0., transition_contract=None):
        self.bundle = Path(bundle)
        self.model_dir = Path(model_dir) if model_dir else find_model()
        self.model = m = mujoco.MjModel.from_xml_path(str(self.model_dir/'mjcf/scene.xml'))
        if abs(m.opt.timestep-.002)>1e-12:
            raise ValueError('Source timestep must be 2 ms; model is never modified')
        self.data = mujoco.MjData(m)
        self.frame = frame
        self.transition_contract=transition_contract
        self.position_names = ['head','left_ear','right_ear','left_frame','right_frame']
        self.rear_names = ['left_rear_wheel','right_rear_wheel']
        self.front_names = ['left_front_wheel','right_front_wheel']
        self.root = m.body('base_link').id
        self.front = [m.body(n).id for n in self.front_names]
        self.rear = [m.body(n).id for n in self.rear_names]
        self.other = [i for i in range(1,m.nbody) if i not in self.rear]
        self.body = [i for i in self.other if i not in self.front]
        self.qpos_ids = [m.joint(n).qposadr[0] for n in self.position_names]
        self.qvel_ids = [m.joint(n).dofadr[0] for n in self.position_names]
        self.wheel_ids = [m.joint(n).dofadr[0] for n in self.rear_names]
        self.actuator_ids = [int(np.where(m.actuator_trnid[:,0]==m.joint(n).id)[0][0])
                             for n in self.position_names+self.rear_names]
        self.contact_history = deque(maxlen=30)
        self.model_sha256 = fingerprint(self.model_dir)
        self.reset()
        manifest = self.controller.manifest
        if self.model_sha256 != manifest['contract']['source_sha256']:
            raise ValueError('MJCF/meshes differ from the exported training source')
        self.spec = manifest['actuation']
        self.kp = np.array([self.spec['position_kp'][n] for n in self.position_names])
        self.kd = np.array([self.spec['position_kd'][n] for n in self.position_names])
        self.effort = np.array([self.spec['drives'][n]['effort'] for n in self.position_names])
        for i,n in enumerate(self.position_names+self.rear_names):
            aid = self.actuator_ids[i]
            if not np.allclose(m.actuator_ctrlrange[aid],self.spec['drives'][n]['control_range']):
                raise ValueError(f'Actuator range mismatch: {n}')

    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
        self.substep = 0
        self.contact_history.clear()
        for _ in range(30):
            self.contact_history.append(np.zeros((self.model.nbody,3)))
        self.controller = Controller(self.bundle,self.state(),self.frame,self.transition_contract)

    def sample_contact(self):
        m,d = self.model,self.data
        net = np.zeros((m.nbody,3))
        force = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m,d,i,force)
            world = c.frame.reshape(3,3).T @ force[:3]
            net[m.geom_bodyid[c.geom1]] -= world
            net[m.geom_bodyid[c.geom2]] += world
        self.contact_history.append(net)

    def velocity(self, body):
        result = np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,body,result,0)
        return np.r_[result[3:],result[:3]]

    def state(self):
        m,d = self.model,self.data
        w,x,y,z = d.xquat[self.root]
        quat = batch([x,y,z,w])
        vel = batch(self.velocity(self.root))
        ground,yaw,rate = wheel_heading(quat,vel)
        amplitude = np.linalg.norm([1-2*(y*y+z*z),2*(x*y+w*z)])
        h = np.array(self.contact_history)
        f = h[-1]
        rear = (h[:,self.rear,2]>.05).all(-1)
        front = (h[:,self.front,2]>.05).all(-1)
        s = dict(quat=quat, pos=batch(d.xpos[self.root]),vel=vel,
            g=batch([2*(w*y-x*z),-2*(w*x+y*z),2*(x*x+y*y)-1]),
            heading=ground*amplitude,yaw=yaw,heading_rate=rate,
            forward=(vel[:,:2]*ground).sum(-1),
            q=batch(d.qpos[self.qpos_ids]),qd=batch(d.qvel[self.qvel_ids]),
            wheel=batch(d.qvel[self.wheel_ids]),qfull=batch(d.qpos[7:]),
            front_z=batch(d.xpos[self.front,2].min()),rear_z=batch(d.xpos[self.rear,2].max()),
            rear=batch(rear[-1],torch.bool),front=batch((np.linalg.norm(f[self.front],axis=-1)>.1).any(),torch.bool),
            other=batch((np.linalg.norm(f[self.other],axis=-1)>.1).any(),torch.bool),
            body_contact=batch((np.linalg.norm(f[self.body],axis=-1)>.1).any(),torch.bool),
            four=batch(rear[-1]&front[-1],torch.bool),rear_fraction=batch(rear.mean()),
            four_fraction=batch((rear&front).mean()),
            other_history=batch((np.linalg.norm(h[:,self.other],axis=-1)>.1).any(),torch.bool),
            body_history=batch((np.linalg.norm(h[:,self.body],axis=-1)>.1).any(),torch.bool),
            front_force_peak=batch(np.linalg.norm(h[:,self.front],axis=-1).max()),
            front_contact=batch((h[:,self.front,2]>.05).any(),torch.bool),
            front_vertical_speed=batch(min(self.velocity(i)[2] for i in self.front)))
        return s

    def servo(self, position, speed):
        d = self.data
        torque = self.kp*(position-d.qpos[self.qpos_ids])-self.kd*d.qvel[self.qvel_ids]
        d.ctrl[self.actuator_ids[:5]] = np.clip(torque,-self.effort,self.effort)
        # Native velocity actuators retain the source kv and effort limit.
        d.ctrl[self.actuator_ids[5:]] = speed

    def step(self):
        for _ in range(30):
            if not self.physics_step():
                return False
        return True

    def physics_step(self):
        """Advance 2 ms; retain the exact 60/20/2 ms controller clocks."""
        c = self.controller
        if c.failure_code:
            return False  # Explicit diagnostic pause; R is the only physical reset.
        if self.substep == 0:
            c.begin(self.state())
        position,speed = c.physics_targets(self.state())
        self.servo(position,speed)
        mujoco.mj_step(self.model,self.data)
        self.sample_contact()
        self.substep = (self.substep+1)%30
        if self.substep == 0:
            c.end(self.state(),float(self.data.time))
        return not bool(c.failure_code)

    def record(self):
        c,p,s = self.controller,self.controller.progress,self.state()
        return dict(time=float(self.data.time),phase=int(p.phase[0]),phase_time=float(p.phase_time[0]),
            armed=c.armed,failure_code=c.failure_code,cycles=c.cycles,
            interruptions=c.interruptions,last_exit_code=c.last_exit_code,
            position=s['pos'][0].tolist(),gravity=s['g'][0].tolist(),
            front_z=float(s['front_z'][0]),rear_fraction=float(s['rear_fraction'][0]),
            four_fraction=float(s['four_fraction'][0]),raw=p.raw[0].tolist(),command=p.command[0].tolist(),
            requested=(p.requested if c.full_range else p.raw)[0].tolist(),
            motion=p.motion[0].tolist(),interval_motion=p.interval_motion[0].tolist(),yaw=float(s['yaw'][0]),
            instant=[float(s['forward'][0]),float(s['heading_rate'][0])],
            frame=s['q'][0,3:].tolist(),frame_target=c.action.target[0,3:].tolist(),
            wheel_measured=s['wheel'][0].tolist(),wheel_target=c.action.speed[0].tolist(),actuator_force=self.data.actuator_force.tolist())
