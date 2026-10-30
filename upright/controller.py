"""One explicit invocation, shared sensor contract and source-limited targets."""
from enum import IntEnum
import torch
from .geometry import wrap

class Phase(IntEnum):
    ORDINARY = 0
    RISE = 1
    BALANCE = 2
    LAND = 3
    HANDOFF = 4  # ordinary controller already owns the actuators
    EXIT = 5     # ordinary owner; physical posture can still be overturned

class Reason(IntEnum):
    NONE = 0
    FALL = 1
    FRONT_RETURN = 2
    RISE_TIMEOUT = 3
    LAND_TIMEOUT = 4
    HANDOFF_TIMEOUT = 5
    INVALID = 6
    WARMUP_TIMEOUT = 7
    SUPPORT_MISMATCH = 8


def slew(current, target, step):
    return current + (target-current).clamp(-step, step)


def wheel_mix(common, differential, limit):
    pair = torch.stack((common-differential, common+differential), -1)
    return pair / (pair.abs().amax(-1, keepdim=True)/limit).clamp_min(1.)


class Controller:
    def __init__(self, n, device, dt, cfg, model, track):
        self.cfg, self.dt, self.n = cfg, dt, n
        self.lo, self.hi = model['joints']['left_frame']['range']
        if model['joints']['right_frame']['range'] != [self.lo, self.hi]:
            raise ValueError('Common hips require matching source limits')
        self.limit = min(50., *(min(abs(x) for x in model['actuators'][k]['control_range'])
                               for k in ('left_rear_wheel', 'right_rear_wheel')))
        self.track = track
        z = lambda: torch.zeros(n, device=device)
        self.phase = z().long(); self.reason = z().long()
        self.age = z(); self.support_run = z(); self.return_run = z(); self.ground_run = z()
        self.lift_run = z(); self.handoff_run = z(); self.yaw_reference = z()
        self.lifted = z().bool(); self.entered = z().bool(); self.completed = z().bool()
        self.command = torch.zeros(n, 2, device=device)
        self.transition_motion = self.command.clone()
        self.ordinary_hip = z()
        self.executed = torch.zeros(n, 3, device=device)  # physical hip, left wheel, right wheel
        self.previous = self.executed.clone(); self.handoff_start = self.executed.clone()
        self.requested_action = self.executed.clone()
        self.requested_command = self.command.clone()
        self.yaw_correction = z()

    def reset(self, ids):
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    def set_command(self, command, hip=None, ids=None):
        ids = slice(None) if ids is None else ids
        command = torch.as_tensor(command, device=self.command.device, dtype=self.command.dtype)
        if not torch.isfinite(command).all():
            raise ValueError('Non-finite command')
        self.command[ids] = command
        if hip is not None:
            target = torch.as_tensor(hip, device=self.command.device, dtype=self.command.dtype)
            if not torch.isfinite(target).all() or (target < self.lo).any() or (target > self.hi).any():
                raise ValueError('Ordinary hip command outside source range')
            ordinary = (self.phase[ids] == Phase.ORDINARY) | (self.phase[ids] == Phase.EXIT)
            self.ordinary_hip[ids] = torch.where(ordinary, target, self.ordinary_hip[ids])

    def request_rise(self, state, ids):
        allowed = (self.phase[ids] == Phase.ORDINARY) | (self.phase[ids] == Phase.EXIT)
        ids = ids[allowed]
        self.phase[ids] = Phase.RISE; self.reason[ids] = Reason.NONE
        for value in (self.age, self.support_run, self.return_run, self.ground_run,
                      self.lift_run, self.handoff_run, self.lifted, self.entered, self.completed):
            value[ids] = 0
        self.yaw_reference[ids] = state['yaw'][ids]
        self.transition_motion[ids] = state['motion'][ids]
        self.transition_motion[ids, 1] = 0
        # Physical state and last executed actuator targets are deliberately retained.

    def request_land(self, state, ids):
        ids = ids[self.phase[ids] == Phase.BALANCE]
        self.phase[ids] = Phase.LAND; self.age[ids] = 0
        self.ground_run[ids] = 0
        self.yaw_reference[ids] = state['yaw'][ids]
        self.transition_motion[ids] = state['motion'][ids]
        self.transition_motion[ids, 1] = 0

    def update(self, s):
        before = self.phase.clone()
        active = (before >= Phase.RISE) & (before <= Phase.HANDOFF)
        self.age += active*self.dt
        finite = torch.isfinite(torch.cat((s['g'], s['gyro'], s['q'], s['motion'], s['gap']), -1)).all(-1)
        fallen = (s['g'][:, 2] > .2) | (s['g'][:, 1].abs() > .8) | (s['gyro'].norm(dim=-1) > 30)
        upright = (s['g'][:, 2] < -.25) & (s['g'][:, 1].abs() < .35)
        support = upright & (s['gap'].amin(-1) > self.cfg.support_gap_m) & (s['rear_skew'].abs() < .03)
        ground = upright & (s['gap'].abs().amax(-1) < self.cfg.ground_gap_m) & (s['rear_skew'].abs() < .02)
        rise_balance = (before == Phase.RISE) | (before == Phase.BALANCE)
        lift = s['gap'].amin(-1) > self.cfg.lift_gap_m
        self.lift_run.copy_(torch.where(rise_balance & lift, self.lift_run+self.dt, 0.))
        self.lifted |= rise_balance & (self.lift_run >= self.cfg.lifted_confirm_s)
        self.return_run.copy_(torch.where(rise_balance & self.lifted & (s['gap'].amin(-1) < self.cfg.ground_gap_m), self.return_run+self.dt, 0.))
        self.support_run.copy_(torch.where(support & rise_balance, self.support_run+self.dt, 0.))
        self.ground_run.copy_(torch.where(ground & (before == Phase.LAND), self.ground_run+self.dt, 0.))
        enter = (before == Phase.RISE) & (self.support_run >= self.cfg.support_enter_s)
        handoff = (before == Phase.LAND) & (self.ground_run >= self.cfg.ground_confirm_s)
        failure = active & (~finite | fallen)
        reason = torch.where(~finite, int(Reason.INVALID), int(Reason.FALL))
        for mask, code in ((rise_balance & (self.return_run >= self.cfg.return_confirm_s), Reason.FRONT_RETURN),
                           ((before == Phase.RISE) & (self.age >= self.cfg.rise_timeout_s), Reason.RISE_TIMEOUT),
                           ((before == Phase.LAND) & (self.age >= self.cfg.landing_timeout_s), Reason.LAND_TIMEOUT),
                           ((before == Phase.HANDOFF) & (self.age >= self.cfg.handoff_timeout_s), Reason.HANDOFF_TIMEOUT)):
            reason = torch.where(mask & ~failure, int(code), reason)
            failure |= mask
        enter &= ~failure; handoff &= ~failure
        self.phase[enter] = Phase.BALANCE; self.age[enter] = 0; self.entered |= enter
        self.phase[handoff] = Phase.HANDOFF; self.age[handoff] = 0
        transferring = handoff | failure
        self.handoff_start[transferring] = self.executed[transferring]
        # Take over the real motion, not the former remote setpoint or zero speed.
        self.command[transferring] = s['motion'][transferring]
        self.ordinary_hip[handoff] = 0
        self.ordinary_hip[failure] = s['q'][failure, 3:].mean(-1).clamp(self.lo, self.hi)
        good_handoff = (before == Phase.HANDOFF) & ground & (s['q'][:, 3:].abs().amax(-1) < .1)
        self.handoff_run.copy_(torch.where(good_handoff, self.handoff_run+self.dt, 0.))
        success = (self.handoff_run >= self.cfg.handoff_hold_s) & ~failure & ~self.completed
        self.completed |= success
        self.phase[success] = Phase.ORDINARY
        self.phase[failure] = Phase.EXIT; self.reason[failure] = reason[failure]
        self.age[failure] = 0
        return dict(phase_before=before, support_estimate=support, ground_estimate=ground,
                    entered_balance=enter, entered_handoff=handoff, success=success, failure=failure,
                    reason=self.reason.clone())

    def targets(self, action, s):
        if action.shape != (self.n, 3) or not torch.isfinite(action).all() or (action.abs() > 1.00001).any():
            raise ValueError('Policy requires three finite bounded actions')
        self.previous.copy_(self.executed); self.requested_action.copy_(action)
        self.requested_command.copy_(self.command)
        policy = (self.phase >= Phase.RISE) & (self.phase <= Phase.LAND)
        transition = (self.phase == Phase.RISE) | (self.phase == Phase.LAND)
        hip = self.lo + (action[:, 0]+1)*.5*(self.hi-self.lo)
        common = action[:, 1]*self.limit
        differential = action[:, 2]*self.limit
        # Deterministic command/measurement feedback, not a parent policy. The
        # one scratch actor still controls the hips, common drive and residual.
        balance_correction = (self.cfg.balance_yaw_feedforward*self.command[:, 1]*self.track/(2*.03)
                              + self.cfg.balance_yaw_feedback*(self.command[:, 1]-s['motion'][:, 1]))
        bound = self.cfg.balance_yaw_correction_limit_rad_s
        self.yaw_correction.copy_(torch.where(self.phase == Phase.BALANCE,
                                              balance_correction.clamp(-bound, bound), 0.))
        differential = (differential+self.yaw_correction).clamp(-self.limit, self.limit)
        # Zero commanded turning during transitions; small feedback resists yaw drift.
        yaw_error = wrap(self.yaw_reference-s['yaw'])
        correction = (self.cfg.heading_kp*yaw_error-self.cfg.heading_kd*s['motion'][:, 1]).clamp(
            -self.cfg.heading_wheel_correction_rad_s, self.cfg.heading_wheel_correction_rad_s)
        differential = torch.where(transition, correction, differential)
        wheels = wheel_mix(common, differential, self.limit)
        rising = self.phase == Phase.RISE
        hip_step = torch.where(rising, self.cfg.rise_hip_slew_rad_s, self.cfg.hip_slew_rad_s)*self.dt
        wheel_step = torch.where(rising, self.cfg.rise_wheel_slew_rad_s2, self.cfg.wheel_slew_rad_s2)*self.dt
        hip = slew(self.executed[:, 0], hip, hip_step)
        wheels = slew(self.executed[:, 1:], wheels, wheel_step[:, None])
        drive = wheel_mix(self.command[:, 0]/.03, self.command[:, 1]*self.track/(2*.03), self.limit)
        ordinary_hip = self.ordinary_hip
        transferring = (self.phase == Phase.HANDOFF) | (self.phase == Phase.EXIT)
        # First ordinary sample inherits the executed targets; subsequent samples blend.
        alpha = (self.age/self.cfg.handoff_blend_s).clamp(0, 1)
        drive = torch.where(transferring[:, None], self.handoff_start[:, 1:] + alpha[:, None]*(drive-self.handoff_start[:, 1:]), drive)
        ordinary_hip = torch.where(transferring, slew(self.executed[:, 0], ordinary_hip, self.cfg.hip_slew_rad_s*self.dt), ordinary_hip)
        self.executed[:, 0].copy_(torch.where(policy, hip, ordinary_hip).clamp(self.lo, self.hi))
        self.executed[:, 1:].copy_(torch.where(policy[:, None], wheels, drive).clamp(-self.limit, self.limit))
        # EXIT is not an active policy stage; advance its handoff blend without re-entering.
        self.age += (self.phase == Phase.EXIT)*self.dt
        return self.executed
