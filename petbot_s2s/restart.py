"""Single-actor restart policy adapter, independent of the legacy C58 runtime."""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import onnxruntime as ort
import torch

from .simulation import Simulation as LegacySimulation, batch, find_model, fingerprint
from .restart_core.controller import Controller as SourceController, Phase, Reason
from .restart_core.controller_v26 import Controller as Round26Controller
from .restart_core.geometry import WheelGeometry, orientation
from .restart_core.observation import Observation


class RestartController:
    full_range = True
    capability_v25 = False
    transition_contract = 'restart_single_actor'
    failure_code = 0  # Expected call interruptions return control; they do not pause physics.

    def __init__(self, bundle, frame):
        self.manifest = json.loads((bundle/'manifest.json').read_text())
        m = self.manifest
        if m['schema'] != 'upright_restart_onnx_v1':
            raise ValueError('Expected an explicitly exported restart ONNX bundle')
        self.cfg = SimpleNamespace(**m['control_config'])
        self.dt = m['control_period_s']
        self.geometry = WheelGeometry(m['wheel_geometry'], 'cpu')
        self.controller_contract = m.get('controller_contract', 'restart_v20')
        controllers = {'restart_v20': SourceController, 'restart_v26': Round26Controller}
        if self.controller_contract not in controllers:
            raise ValueError(f'Unknown restart controller contract: {self.controller_contract}')
        self.control = controllers[self.controller_contract](
            1, 'cpu', self.dt, self.cfg, m['source_model'], self.geometry.track)
        if not self.control.lo <= frame <= self.control.hi or not math.isfinite(frame):
            raise ValueError('Initial hip target is outside the source range')
        self.frame = float(frame)
        self.frame_input = 0.
        self.control.ordinary_hip[:] = frame
        self.observation = Observation()
        self.observation.cfg, self.observation.control = self.cfg, self.control
        self.observation.probe_level = 1  # deterministic inference: no training observation noise
        self.history = deque(maxlen=self.cfg.history)
        self.speed_limit, self.yaw_limit = m['policy_limits']['speed'], m['policy_limits']['yaw']
        self.ordinary = SimpleNamespace(speed_limit=.03*self.control.limit,
                                       yaw_limit=.06*self.control.limit/self.geometry.track)
        self.policy_motion_limits = torch.tensor([self.speed_limit, self.yaw_limit])
        self.operator_command = torch.zeros(1, 2)
        self.raw = self.operator_command.clone()
        self.armed = False
        self.pending_rise = self.pending_land = False
        self.cycles = self.interruptions = self.last_exit_code = 0
        self.events = []
        self.elapsed = 0.
        self.balance_seen = False
        self.progress = SimpleNamespace(phase=self.control.phase, phase_time=self.control.age)
        self.last_action = torch.zeros(1, 3)
        self.last_observation = None
        self.position = np.array([m['source_model']['joints'][n]['default'] for n in m['actuation']['position_joints']])
        self.speed = np.zeros(2)
        self.inference_count = 0
        path = bundle/m['model']
        if hashlib.sha256(path.read_bytes()).hexdigest() != m['networks']['policy']['sha256']:
            raise ValueError('ONNX fingerprint mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.network = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        if self.network.get_inputs()[0].shape[-1] != m['frame_dim']*self.cfg.history:
            raise ValueError('Observation dimensions do not match the bundle')

    @property
    def manual_frame_allowed(self):
        return not self.armed and int(self.control.phase[0]) in (Phase.ORDINARY, Phase.EXIT)

    @property
    def ordinary_control(self):
        return int(self.control.phase[0]) in (Phase.ORDINARY, Phase.HANDOFF, Phase.EXIT)

    def set_policy_limits(self, speed, yaw):
        if not all(math.isfinite(v) and 0 <= v <= limit+1e-7 for v, limit in
                   ((speed, self.speed_limit), (yaw, self.yaw_limit))):
            raise ValueError('Policy limits outside exported command range')
        self.policy_motion_limits[:] = torch.tensor([speed, yaw])

    def set_command(self, speed, yaw):
        if not all(math.isfinite(v) and abs(v) <= limit+1e-7 for v, limit in
                   ((speed, self.ordinary.speed_limit), (yaw, self.ordinary.yaw_limit))):
            raise ValueError('Operator command outside source range')
        command = torch.tensor([[speed, yaw]], dtype=torch.float32)
        self.operator_command.copy_(command)

    def set_frame_input(self, direction):
        if not math.isfinite(direction):
            raise ValueError('Invalid frame input')
        self.frame_input = min(1., max(-1., direction)) if self.manual_frame_allowed else 0.

    def toggle(self):
        if self.pending_rise:
            self.pending_rise = self.armed = False
        elif self.manual_frame_allowed:
            self.pending_rise = self.armed = True
            self.frame_input = 0.
        elif int(self.control.phase[0]) in (Phase.RISE, Phase.BALANCE):
            self.pending_land = True  # latch during RISE, apply when BALANCE starts

    def begin(self, state):
        c = self.control
        if self.pending_rise:
            c.request_rise(state, torch.tensor([0]))
            self.pending_rise = False
            self.pending_land = False
            self.history.clear()
            self.balance_seen = False
            self.events.append(dict(time=self.elapsed, event='rise_requested'))
        if self.pending_land and int(c.phase[0]) == Phase.BALANCE:
            c.request_land(state, torch.tensor([0]))
            self.pending_land = False
            self.events.append(dict(time=self.elapsed, event='land_requested'))
        if self.ordinary_control:
            # HANDOFF already belongs to ordinary control. Its pinned target blend
            # inherits the last wheel targets and approaches the live request;
            # hip/ground confirmation must not latch measured landing velocity.
            command = self.operator_command
        else:
            command = torch.maximum(torch.minimum(self.operator_command, self.policy_motion_limits), -self.policy_motion_limits)
        self.raw.copy_(command)
        c.set_command(command, hip=self.frame)
        c.update_motion_reference()
        frame = self.observation.observation(state)
        if not self.history:
            self.history.extend(frame.clone() for _ in range(self.cfg.history))
        else:
            self.history.append(frame.clone())
        self.last_observation = torch.cat(tuple(self.history), dim=-1)
        if int(c.phase[0]) in (Phase.RISE, Phase.BALANCE, Phase.LAND):
            output = self.network.run(None, {'obs': self.last_observation.numpy()})[0]
            self.last_action = torch.from_numpy(output)
            self.inference_count += 1
        else:
            self.last_action.zero_()
        targets = c.targets(self.last_action, state)[0].numpy()
        self.position[3:] = targets[0]
        self.speed = targets[1:].copy()

    def end(self, state, time, body_contact, root_height):
        c = self.control
        self.elapsed = time
        before = int(c.phase[0])
        result = c.update(state)
        active = before in (Phase.RISE, Phase.BALANCE, Phase.LAND, Phase.HANDOFF)
        # Simulator contact is an independent safety/validation signal, never an actor input.
        physical_fall = active and (body_contact or root_height < .025)
        if physical_fall and not bool(result['failure'][0]):
            c.handoff_start.copy_(c.executed)
            c.command.copy_(state['motion'])
            c.ordinary_hip[:] = state['q'][:, 3:].mean(-1).clamp(c.lo, c.hi)
            c.phase[:] = Phase.EXIT
            c.reason[:] = Reason.FALL
            c.age.zero_()
            result['failure'][:] = True
        if bool(result['entered_balance'][0]):
            self.balance_seen = True
            self.events.append(dict(time=time, event='balance_entered'))
        if bool(result['entered_handoff'][0]) or bool(result['failure'][0]):
            self.frame = float(c.ordinary_hip[0])
        if bool(result['failure'][0]):
            # EXIT already blends from executed targets in the pinned controller.
            self.interruptions += 1
            self.last_exit_code = int(c.reason[0])
            self.armed = self.pending_land = False
            self.frame_input = 0.
            self.events.append(dict(time=time, event='call_interrupted', reason=Reason(self.last_exit_code).name))
        elif bool(result['success'][0]):
            self.cycles += 1
            self.armed = self.pending_land = False
            self.frame = 0.
            self.events.append(dict(time=time, event='ordinary_handoff_complete'))


class RestartSimulation(LegacySimulation):
    def __init__(self, bundle, model_dir=None, frame=0., transition_contract=None):
        if transition_contract is not None:
            raise ValueError('Legacy controller interventions cannot be applied to restart policies')
        self.bundle = Path(bundle)
        self.model_dir = Path(model_dir) if model_dir else find_model()
        self.model = m = mujoco.MjModel.from_xml_path(str(self.model_dir/'mjcf/scene.xml'))
        self.data = mujoco.MjData(m)
        self.frame = frame
        self.position_names = ['head', 'left_ear', 'right_ear', 'left_frame', 'right_frame']
        self.rear_names = ['left_rear_wheel', 'right_rear_wheel']
        self.front_names = ['left_front_wheel', 'right_front_wheel']
        self.root = m.body('base_link').id
        self.front = [m.body(n).id for n in self.front_names]
        self.rear = [m.body(n).id for n in self.rear_names]
        self.body = [i for i in range(1, m.nbody) if i not in self.front+self.rear]
        self.qpos_ids = [m.joint(n).qposadr[0] for n in self.position_names]
        self.qvel_ids = [m.joint(n).dofadr[0] for n in self.position_names]
        self.wheel_ids = [m.joint(n).dofadr[0] for n in self.rear_names]
        self.actuator_ids = [int(np.where(m.actuator_trnid[:, 0] == m.joint(n).id)[0][0])
                             for n in self.position_names+self.rear_names]
        self.model_sha256 = fingerprint(self.model_dir)
        manifest = json.loads((self.bundle/'manifest.json').read_text())
        self.decimation = manifest['control_config']['decimation']
        if self.model_sha256 != manifest['metadata']['model_sha256'] or abs(m.opt.timestep-.002) > 1e-12:
            raise ValueError('Source model/timestep mismatch')
        if abs(self.decimation*m.opt.timestep-manifest['control_period_s']) > 1e-12:
            raise ValueError('Exported policy period does not match decimation')
        self.contact_history = deque(maxlen=self.decimation)
        self.spec = manifest['actuation']
        self.kp = np.array([self.spec['position_kp'][n] for n in self.position_names])
        self.kd = np.array([self.spec['position_kd'][n] for n in self.position_names])
        self.effort = np.array([self.spec['drives'][n]['effort'] for n in self.position_names])
        for i, n in enumerate(self.position_names+self.rear_names):
            if not np.allclose(m.actuator_ctrlrange[self.actuator_ids[i]], self.spec['drives'][n]['control_range']):
                raise ValueError('Actuator limits differ from the training source')
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.substep = 0
        self.contact_history.clear()
        self.contact_history.extend(np.zeros((self.model.nbody, 3)) for _ in range(self.decimation))
        self.controller = RestartController(self.bundle, self.frame)

    def state(self):
        d = self.data
        w, x, y, z = d.xquat[self.root]
        velocity = self.velocity(self.root)
        g, yaw, gyro, motion = orientation(batch([x, y, z, w]), batch(velocity[:3]), batch(velocity[3:]))
        q = batch(d.qpos[self.qpos_ids])
        gap, skew = self.controller.geometry.estimate(g, q[:, 3:])
        return dict(g=g, yaw=yaw, gyro=gyro, motion=motion, q=q,
                    qd=batch(d.qvel[self.qvel_ids]), wheel=batch(d.qvel[self.wheel_ids]), gap=gap, rear_skew=skew)

    def step(self):
        for _ in range(self.decimation):
            self.physics_step()
        return True

    def physics_step(self):
        c = self.controller
        if c.manual_frame_allowed:
            c.frame = min(c.control.hi, max(c.control.lo, c.frame+.002*.5*c.frame_input))
        if self.substep == 0:
            c.begin(self.state())
        self.servo(c.position, c.speed)
        mujoco.mj_step(self.model, self.data)
        self.sample_contact()
        self.substep = (self.substep+1) % self.decimation
        if self.substep == 0:
            h = np.array(self.contact_history)
            contact = bool((np.linalg.norm(h[:, self.body], axis=-1) > .1).any())
            c.end(self.state(), float(self.data.time), contact, float(self.data.xpos[self.root, 2]))
        return True

    def record(self):
        c, s = self.controller, self.state()
        p = c.control
        h = np.array(self.contact_history)
        rear = (h[:, self.rear, 2] > .05).all(-1)
        front = (h[:, self.front, 2] > .05).all(-1)
        return dict(time=float(self.data.time), phase=int(p.phase[0]), phase_time=float(p.age[0]),
                    armed=c.armed, failure_code=0, cycles=c.cycles, interruptions=c.interruptions,
                    last_exit_code=c.last_exit_code, inference_backend='onnxruntime_cpu',
                    operator_request=c.operator_command[0].tolist(), ordinary_control=c.ordinary_control,
                    ordinary_frame_target=c.frame, manual_frame_allowed=c.manual_frame_allowed,
                    policy_motion_limits=c.policy_motion_limits.tolist(), position=self.data.xpos[self.root].tolist(),
                    gravity=s['g'][0].tolist(), front_z=float(self.data.xpos[self.front, 2].min()),
                    rear_z=float(self.data.xpos[self.rear, 2].max()), rear_fraction=float(rear.mean()),
                    four_fraction=float((rear & front).mean()), raw=c.raw[0].tolist(),
                    command=p.effective_command[0].tolist(), requested=p.requested_command[0].tolist(),
                    control_reference=p.executed_reference[0].tolist(), motion=s['motion'][0].tolist(),
                    interval_motion=s['motion'][0].tolist(), yaw=float(s['yaw'][0]), instant=s['motion'][0].tolist(),
                    frame=s['q'][0, 3:].tolist(), frame_target=c.position[3:].tolist(),
                    wheel_measured=s['wheel'][0].tolist(), wheel_target=c.speed.tolist(),
                    actuator_force=self.data.actuator_force.tolist(), inference_count=c.inference_count)
