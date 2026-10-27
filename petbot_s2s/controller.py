"""Portable controller: exact tensor core plus operator-controlled admission."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .core.action_terms import FeedbackAction, attach_balance_reflex
from .core.continuous import ContinuousProgress, bound_cycle_features
from .core.local_frame import ground_frame_features


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
        assert c['yaw_feedback_design'] == 'fast2' and c['balance_reflex'] == 'frozen20'
        assert c['handoff_alignment']['mode'] == 'after'
        self.actor = torch.jit.load(str(bundle/'actor.pt')).eval()
        self.reflex = torch.jit.load(str(bundle/'balance_reflex.pt')).eval()
        self.spec = self.manifest['actuation']
        self.num_envs, self.device, self.step_dt, self.physics_dt = 1, 'cpu', .06, .002
        self.cfg = SimpleNamespace(steering_design='feedback', yaw_filter_tau=c['yaw_filter_tau'],
            frame_slew=c['frame_slew'], wheel_slew=c['wheel_slew'],
            yaw_feedback_design='fast2', yaw_feedback_gain=c['fast_yaw_controller']['kp'])
        self.cfg.transition_contract=self.transition_contract
        limits = self.manifest['tested_range']
        self.speed_limit, self.yaw_limit = limits['speed_limit'], limits['yaw_limit']
        self.frame_range = (limits['frame_min'], limits['frame_max'])
        if not self.frame_range[0] <= frame <= self.frame_range[1]:
            raise ValueError(f'Frame outside exported envelope {self.frame_range}')
        self.frame = frame
        self.progress = ContinuousProgress(1, 'cpu', .06, self.speed_limit, self.yaw_limit,
            self.frame_range, 42, 'pose_feedback_tight', hold_range=(180.,180.),
            tracking_objective='stability_first', handoff_alignment='after',
            transition_contract=self.transition_contract)
        self.robot = RobotTargets(self.spec)
        self.scene = {'robot': self.robot}
        self.action = FeedbackAction(None, self)
        self.action_manager = SimpleNamespace(get_term=lambda name: self.action)
        attach_balance_reflex(self, SimpleNamespace(balance_reflex=self.reflex), True)
        self.action.reset()
        self.s = initial_state
        self.history = None
        self.armed = False
        self.failure_code = 0
        self.cycles = 0
        self.events = []
        self.last_metrics = {}
        self.reset_progress(initial_state, preserve=False)

    def reset_progress(self, s, preserve=True):
        """Re-arm after handoff, preserving physical state, targets and motion."""
        p = self.progress
        command, raw, motion = p.command.clone(), p.raw.clone(), p.motion.clone()
        p.reset(torch.tensor([0]), s['pos'], s['yaw'])
        p.frame[:] = self.frame
        p.request_mode(0, self.frame)
        p.switch_at[:] = 1e9
        p.set_command(raw if preserve else [0., 0.])
        if preserve:
            p.command.copy_(command)
            p.motion.copy_(motion)
        self.armed = False
        self.history = None

    def set_command(self, forward, yaw):
        self.progress.set_command([forward, yaw])

    def toggle(self):
        if self.failure_code:
            return
        p = self.progress
        phase = int(p.phase[0])
        if phase == 0 and not self.armed:
            self.armed = True
            p.age[:] = 0.
            p.switch_at[:] = 0.
            p.request_mode(1, self.frame)
        elif phase < 3:
            p.request_mode(0, self.frame)
            if phase == 0:  # Cancel a still-pending admission.
                self.armed = False
                p.switch_at[:] = 1e9

    def features(self):
        s, p, a = self.s, self.progress, self.action
        base = torch.cat((s['g'],s['heading'],s['vel'],s['q'],s['qd']*.1,s['wheel']*.05,
            p.raw,p.command,s['pos'][:,:2]-p.reference,s['pos'][:,:2]-p.raw_reference,
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
        self.progress.before_step()
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
        if bool(m['failure'][0]):
            self.failure_code = int(m['failure_code'][0])
            self.events.append(dict(time=time,failure_code=self.failure_code))
        elif bool(m['completed'][0]):
            self.cycles += 1
            self.events.append(dict(time=time,completed_cycle=self.cycles))
            self.reset_progress(s)
