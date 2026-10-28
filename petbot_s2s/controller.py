"""Portable controller: exact tensor core plus operator-controlled admission."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from .core.action_terms import FeedbackAction, attach_balance_reflex
from .core.continuous import ContinuousProgress, bound_cycle_features
from .core.local_frame import ground_frame_features
from .core.full_range import FullRangeProgress, effective_request
from .networks import OnnxNetwork
from .ordinary import OrdinaryDrive


class RobotTargets:
    """Named-joint interface required by the vendored action terms."""
    def __init__(self, spec):
        self.names = spec['position_joints'] + spec['velocity_joints']
        defaults = [spec['joints'][n]['default'] for n in self.names]
        self.data = SimpleNamespace(default_joint_pos=torch.tensor([defaults]),
                                    root_link_pose_w=None, root_com_vel_w=None)

    def find_joints(self, names, preserve_order=True):
        return [self.names.index(n) for n in names], names

    def set_joint_position_target(self, value, joint_ids):
        self.position = value.clone()

    def set_joint_velocity_target(self, value, joint_ids):
        self.velocity = value.clone()


class Controller:
    def __init__(self, bundle: Path, initial_state, frame=0., transition_contract=None):
        torch.set_num_threads(1)
        self.bundle = bundle
        self.manifest = json.loads((bundle/'manifest.json').read_text())
        c = self.manifest['contract']
        self.transition_contract=transition_contract or c.get('transition_contract','legacy')
        for name, sha in self.manifest['networks'].items():
            if hashlib.sha256((bundle/name).read_bytes()).hexdigest() != sha:
                raise ValueError(f'Corrupt policy file: {name}')
        assert c['control_dt'] == .06 and c['observation_schema'] == 'cycle_58x5_balance_local_v4'
        assert c['yaw_feedback_design'] == 'fast2' and c['balance_reflex'] in ('frozen20', 'parent_full20')
        self.full_parent_feedback = c['balance_reflex'] == 'parent_full20'
        assert c['handoff_alignment']['mode'] == 'after'
        if self.manifest.get('format') != 'petbot_cycle_onnx_bundle_v2':
            raise ValueError('Convert this policy bundle to ONNX before running sim2sim')
        self.actor = OnnxNetwork(bundle/'actor.onnx')
        self.reflex = OnnxNetwork(bundle/'balance_reflex.onnx')
        self.parent = None
        if self.full_parent_feedback:
            assert c['task'] == 'upright_cycle_v23'
            assert 'parent_balance.onnx' in self.manifest['networks']
            feedback = c['reflex_contract']
            assert feedback['version'] == 'frozen_full_parent_offset_v1'
            assert feedback['period'] == .02 and feedback['gate_after'] == 1.8
            self.parent = OnnxNetwork(bundle/'parent_balance.onnx')
        self.spec = self.manifest['actuation']
        self.ordinary = OrdinaryDrive(self.spec)
        self.operator_command = torch.zeros(1, 2)
        self.frame_input = 0.
        self.num_envs, self.device, self.step_dt, self.physics_dt = 1, 'cpu', .06, .002
        self.cfg = SimpleNamespace(steering_design='feedback', yaw_filter_tau=c['yaw_filter_tau'],
            frame_slew=c['frame_slew'], wheel_slew=c['wheel_slew'],
            yaw_feedback_design='fast2', yaw_feedback_gain=c['fast_yaw_controller']['kp'])
        self.cfg.transition_contract=self.transition_contract
        self.full_range=c.get('task_profile')=='full_range'
        self.cfg.task_profile=c.get('task_profile','legacy')
        limits = self.manifest['tested_range']
        self.speed_limit, self.yaw_limit = limits['speed_limit'], limits['yaw_limit']
        self.policy_motion_limits = torch.tensor([self.speed_limit, self.yaw_limit])
        self.frame_range = (limits['frame_min'], limits['frame_max'])
        if not math.isfinite(frame) or not self.ordinary.frame_range[0] <= frame <= self.ordinary.frame_range[1]:
            raise ValueError(f'Frame outside source joint limits {self.ordinary.frame_range}')
        self.frame = frame
        progress_type=FullRangeProgress if self.full_range else ContinuousProgress
        self.progress = progress_type(1, 'cpu', .06, self.ordinary.speed_limit, self.ordinary.yaw_limit,
            self.ordinary.frame_range, 42, 'pose_feedback_tight', hold_range=(180.,180.),
            tracking_objective='stability_first', handoff_alignment='after',
            transition_contract=self.transition_contract)
        self.robot = RobotTargets(self.spec)
        self.scene = {'robot': self.robot}
        self.action = FeedbackAction(None, self)
        self.action_manager = SimpleNamespace(get_term=lambda name: self.action)
        attach_balance_reflex(self, SimpleNamespace(balance_reflex=self.reflex, parent_balance=self.parent),
                             True, full_parent=self.full_parent_feedback)
        self.action.reset()
        self.s = initial_state
        self.history = None
        self.armed = False
        self.failure_code = 0
        self.cycles = 0
        self.interruptions = 0
        self.last_exit_code = 0
        self.events = []
        self.last_metrics = {}
        self.reset_progress(initial_state, preserve=False)

    @torch.inference_mode()
    def reset_progress(self, s, preserve=True):
        """Re-arm after handoff, preserving physical state, targets and motion."""
        p = self.progress
        command, motion = p.command.clone(), p.motion.clone()
        p.reset(torch.tensor([0]), s['pos'], s['yaw'])
        p.frame[:] = self.frame
        p.request_mode(0, self.ordinary.default_frame)
        p.switch_at[:] = 1e9
        p.set_command(self.operator_command if preserve else [0., 0.])
        if preserve:
            p.command.copy_(command)
            p.motion.copy_(motion)
        self.armed = False
        self.history = None
        if self.full_range and preserve:
            # Clear per-call feedback/history without jumping held motor targets.
            saved={k:getattr(self.action,k).clone() for k in ('target','speed','applied','previous')}
            self.action.reset()
            for k,v in saved.items():getattr(self.action,k).copy_(v)

    @torch.inference_mode()
    def set_command(self, forward, yaw):
        self.ordinary.validate(forward, yaw)
        self.operator_command[:] = self.operator_command.new_tensor([forward, yaw])
        self.route_command()

    @property
    def ordinary_control(self):
        return int(self.progress.phase[0]) in (0, 4)

    @property
    def manual_frame_allowed(self):
        return not self.armed and int(self.progress.phase[0]) == 0

    def set_policy_limits(self, speed, yaw):
        if not all(math.isfinite(x) and x >= 0 for x in (speed, yaw)):
            raise ValueError('Invalid two-wheel command limits')
        if speed > self.speed_limit or yaw > self.yaw_limit:
            raise ValueError('Two-wheel limits exceed the exported policy envelope')
        self.policy_motion_limits[:] = torch.tensor([speed, yaw])

    def set_frame_input(self, direction):
        if not math.isfinite(direction) or abs(direction)>1:
            raise ValueError('Frame input must be in [-1, 1]')
        self.frame_input = float(direction) if self.manual_frame_allowed else 0.

    def route_command(self):
        command = self.operator_command
        if self.armed and int(self.progress.phase[0]) < 4:
            command = command.clamp(min=-self.policy_motion_limits, max=self.policy_motion_limits)
        self.progress.set_command(command)

    @torch.inference_mode()
    def toggle(self):
        if self.failure_code:
            return
        p = self.progress
        phase = int(p.phase[0])
        if phase == 0 and not self.armed:
            if self.full_range:self.reset_progress(self.s)
            self.armed = True
            p.age[:] = 0.
            p.switch_at[:] = 0.
            self.frame_input = 0.
            p.request_mode(1, self.ordinary.default_frame)
        elif phase < 3:
            p.request_mode(0, self.ordinary.default_frame)
            if phase > 0:
                self.frame = self.ordinary.default_frame
            if phase == 0:  # Cancel a still-pending admission.
                self.armed = False
                p.switch_at[:] = 1e9

    def features(self):
        s, p, a = self.s, self.progress, self.action
        base = torch.cat((s['g'],s['heading'],s['vel'],s['q'],s['qd']*.1,s['wheel']*.05,
            effective_request(p) if self.full_range else p.raw,p.command,s['pos'][:,:2]-p.reference,s['pos'][:,:2]-p.raw_reference,
            torch.stack((p.raw_yaw.sin(),p.raw_yaw.cos()),-1),a.applied,
            p.active[:,None].float(),p.since[:,None]/12,p.peak[:,None],
            p.support_run[:,None].clamp(max=1),p.frame[:,None]),-1)
        features = torch.cat((base,torch.nn.functional.one_hot(p.phase,5).float(),
            p.requested_mode[:,None],p.destination[:,None],p.phase_time[:,None]/4),-1)
        initial=p.transition_heading if self.transition_contract=='linear_heading' else None
        old = bound_cycle_features(ground_frame_features(features,s['yaw'],'balance_local',initial))
        instantaneous = torch.stack((s['forward'],s['heading_rate']),-1)
        feedback = torch.cat((instantaneous,p.motion,p.command-p.motion,p.error_integral),-1)
        feedback = feedback/feedback.new_tensor([.1,.3,.1,.3,.1,.3,.2,.6])
        return torch.cat((old,feedback,a.yaw_feedback.integral[:,None]/1.5),-1)

    def set_state(self, s):
        self.s = s
        self.robot.data.root_link_pose_w = torch.cat((s['pos'],s['quat']),-1)
        self.robot.data.root_com_vel_w = s['vel']

    @torch.inference_mode()
    def begin(self, s):
        self.set_state(s)
        self.route_command()
        self.progress.before_step()
        if self.ordinary_control:
            # Four-wheel command acceleration is not the policy's training slew.
            self.progress.command.copy_(self.progress.raw)
        features = self.features()
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite observation')
        if self.history is None:
            self.history = features[:,None,:].repeat(1,5,1)
        else:
            self.history = torch.cat((self.history[:,1:],features[:,None,:]),1)
        raw = self.history.flatten(1)
        self.action.policy_observation = raw
        self.action.process_actions(self.actor(raw))

    @torch.inference_mode()
    def physics_targets(self, s):
        self.set_state(s)
        if self.ordinary_control:
            p, a = self.progress, self.action
            if self.manual_frame_allowed:
                lo, hi = self.ordinary.frame_range
                self.frame = min(hi, max(lo, self.frame+self.frame_input*self.ordinary.frame_input_rate*self.physics_dt))
                p.frame[:] = self.frame
            desired = self.ordinary.default_frame if int(p.phase[0]) == 4 else self.frame
            frame = a.target[:, 3]+(desired-a.target[:, 3]).clamp(
                -self.ordinary.frame_target_slew*self.physics_dt,
                self.ordinary.frame_target_slew*self.physics_dt)
            a.target[:, 3:] = frame[:, None]
            a.speed.copy_(self.ordinary.wheels(p.command, p.motion, self.physics_dt))
            a.applied[:, 0] = 2*(frame-a.lo)/(a.hi-a.lo)-1
            a.applied[:, 1:] = a.speed/a.max_speed
            self.robot.set_joint_position_target(a.target, a.pos_ids)
            self.robot.set_joint_velocity_target(a.speed, a.wheel_ids)
        else:
            self.ordinary.integral.zero_()
            self.action.apply_actions()
        return self.robot.position[0].numpy().copy(), self.robot.velocity[0].numpy().copy()

    @torch.inference_mode()
    def end(self, s, time):
        self.set_state(s)
        p = self.progress
        # Waiting for Space is not the training environment's six-second warmup.
        if not self.armed and not bool(p.active[0]):
            p.age[:] = 0.
        previous = int(p.phase[0])
        p.update_motion(s)
        fixed = s['q'].new_tensor([self.spec['joints'][n]['default'] for n in self.spec['position_joints'][:3]])
        self.last_metrics = m = p.update(s,fixed)
        if int(p.phase[0]) != previous:
            self.events.append(dict(time=time,phase=int(p.phase[0])))
        if self.manual_frame_allowed and bool(m['failure'][0]) and not (int(m['failure_code'][0])&1):
            # Ordinary driving is not an upright episode. Policy posture/speed
            # failure criteria must not pause the ordinary controller.
            return
        if bool(m['failure'][0]) and self.full_range and not (int(m['failure_code'][0])&1):
            # Physical failure ends the call; ordinary control continues in the
            # actual pose, even when overturned. A new Space request is required.
            if self.armed or bool(p.active[0]):
                self.interruptions += 1
                self.last_exit_code = int(m['failure_code'][0])
                self.events.append(dict(time=time,interrupted_call=self.interruptions,
                    failure_code=self.last_exit_code,exit_phase=int(m.get('exit_phase',p.phase)[0]),
                    ordinary_pose=bool(m.get('geometry_ground',s['four'])[0])))
                self.frame = float(self.action.target[0, 3])
                self.reset_progress(s)
        elif bool(m['failure'][0]):
            self.failure_code = int(m['failure_code'][0])
            self.events.append(dict(time=time,failure_code=self.failure_code))
        elif bool(m['completed'][0]):
            self.cycles += 1
            self.events.append(dict(time=time,completed_cycle=self.cycles))
            self.frame = self.ordinary.default_frame
            self.reset_progress(s)
