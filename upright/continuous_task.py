"""Full cycle v7: command timing, continuous holding, explicit motion feedback."""
import torch
from isaaclab.managers import ActionTermCfg, ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup, RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm, EventTermCfg as EventTerm
from isaaclab.utils import configclass
from . import cycle_task as base
from .cycle_task import SPEC, tensor, state, POSITION_JOINTS
from .moving import slew, wheel_targets
from .cycle import filter_differential
from .continuous import ContinuousProgress, yaw_feedforward, bound_cycle_features


def ensure(env):
    if not hasattr(env, 'moving_progress'):
        s = state(env)
        env.moving_progress = ContinuousProgress(env.num_envs, env.device, env.step_dt,
            env.cfg.speed_limit, env.cfg.yaw_limit, env.cfg.frame_range, env.cfg.seed,
            env.cfg.feedback_mode, hold_range=env.cfg.hold_range,
            instant_tracking_weight=getattr(env.cfg,'instant_tracking_weight',.3))
        env.moving_progress.reset(torch.arange(env.num_envs, device=env.device), s['pos'], s['yaw'])
        env.moving_step = -1
    return env.moving_progress


def prepare_commands(env):
    p = ensure(env)
    if p.prepared_step != env.common_step_counter:
        p.before_step()
        p.prepared_step = env.common_step_counter


class ContinuousAction(base.MovingAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.drive_integral = torch.zeros(env.num_envs, 2, device=env.device)

    def process_actions(self, actions):
        if not torch.isfinite(actions).all() or actions.abs().max()>1.00001:
            raise ValueError('Invalid continuous action')
        p = ensure(self._env)
        self._raw.copy_(actions)
        self.previous.copy_(self.applied)
        desired_frame = self.lo+(actions[:, 0]+1)*.5*(self.hi-self.lo)
        desired_speed = wheel_targets(actions, self.max_speed, 'common_diff')
        if self._env.cfg.steering_design == 'residual':
            blend = torch.where(p.phase==2, (p.phase_time/.6).clamp(0, 1),
                    torch.where(p.phase==3, (1.-p.phase_time/.6).clamp(0, 1), 0.))
            difference = (1.-blend)*actions[:, 2]*5.+blend*(yaw_feedforward(p.command[:, 1])+actions[:, 2]*1.5)
            common = actions[:, 1]*self.max_speed
            desired_speed = torch.stack((common-difference, common+difference), -1).clamp(-self.max_speed, self.max_speed)
        ordinary = ~p.active | (p.phase>=4)
        error = p.command-p.motion
        gains = error.new_tensor([20., 8.])
        limit = error.new_tensor([5., 2.])
        integral = self.drive_integral+self._env.step_dt*gains*error
        self.drive_integral[:] = torch.where(ordinary[:, None], torch.maximum(torch.minimum(integral, limit), -limit), 0.)
        common = p.command[:, 0]/.03+10.*error[:, 0]+self.drive_integral[:, 0]
        difference = yaw_feedforward(p.command[:, 1])+2.*error[:, 1]+self.drive_integral[:, 1]
        drive = torch.stack((common-difference, common+difference), -1).clamp(-self.max_speed, self.max_speed)
        desired_speed = torch.where(ordinary[:, None], drive, desired_speed)
        desired_frame = torch.where(~p.active, p.frame, desired_frame)
        desired_frame = torch.where(p.phase>=4, p.destination, desired_frame)
        filtered = filter_differential(self.speed, desired_speed, self._env.step_dt, self._env.cfg.yaw_filter_tau, self.max_speed)
        desired_speed = torch.where(ordinary[:, None], desired_speed, filtered)
        frame = slew(self.target[:, 3], desired_frame, self._env.cfg.frame_slew*self._env.step_dt)
        self.target[:, 3:] = frame[:, None]
        self.speed.copy_(slew(self.speed, desired_speed, self._env.cfg.wheel_slew*self._env.step_dt))
        self.applied[:, 0] = 2*(frame-self.lo)/(self.hi-self.lo)-1
        self.applied[:, 1:] = self.speed/self.max_speed

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.drive_integral[slice(None) if env_ids is None else env_ids] = 0.


def metrics(env):
    p = ensure(env)
    if env.moving_step != env.common_step_counter:
        s = state(env)
        p.update_motion(s)
        fixed = s['q'].new_tensor([SPEC['joints'][n]['default'] for n in POSITION_JOINTS[:3]])
        m = p.update(s, fixed)
        env.moving_metrics = (s, m)
        env.moving_step = env.common_step_counter
        action = env.action_manager.get_term('moving')
        if hasattr(action, 'yaw_feedback'):
            m['yaw_feedback_integral'] = action.yaw_feedback.integral.clone()
            m['yaw_feedback_target'] = action.yaw_feedback.target.clone()
        env.moving_snapshot = ({k:v.clone() for k,v in s.items()}, {k:v.clone() for k,v in m.items()},
                              p.raw.clone(), p.command.clone(), action.applied.clone(), p.since.clone())
        env.extras.setdefault('log', {}).update({'Cycle/active':p.active.float().mean(),
            'Cycle/balance':(p.phase==2).float().mean(), 'Cycle/complete':m['completed'].float().mean(),
            'Cycle/qualified':m['qualified'].float().mean(), 'Cycle/actual_admission':m['motion_attained'].float().mean()})
    return env.moving_metrics


def observations(env):
    prepare_commands(env)
    p = ensure(env)
    s = state(env)
    old = bound_cycle_features(base.observations(env))
    instantaneous = torch.stack((s['forward'], s['heading_rate']), -1)
    feedback = torch.cat((instantaneous, p.motion, p.command-p.motion, p.error_integral), -1)
    feedback = feedback/feedback.new_tensor([.1, .3, .1, .3, .1, .3, .2, .6])
    return torch.cat((old, feedback), -1)


def reward(env, kind):
    metrics(env)
    return base.reward(env, kind)


def failed(env):
    return metrics(env)[1]['failure']


def completed(env):
    return metrics(env)[1]['completed']


def time_out(env):
    p = ensure(env)
    m = metrics(env)[1]
    return p.active & (p.since>=env.cfg.active_seconds) & ~m['failure'] & ~m['completed']


def reset(env, env_ids):
    ensure(env)
    base.reset(env, env_ids)


@configclass
class ActionsCfg:
    moving = ActionTermCfg(class_type=ContinuousAction, asset_name='robot')


@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObsGroup):
        state = ObsTerm(func=observations)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True
    policy = Policy()


@configclass
class RewardsCfg(base.RewardsCfg):
    def __post_init__(self):
        for term in vars(self).values():
            if isinstance(term, RewTerm):
                term.func = reward
        self.tracking.weight = 12.
        self.impact.weight = -.2
        self.descent.weight = -.3
        self.touchdown.weight = -.2


@configclass
class DonesCfg:
    complete = DoneTerm(func=completed)
    timeout = DoneTerm(func=time_out, time_out=True)
    failure = DoneTerm(func=failed)


@configclass
class EventsCfg:
    reset_robot = EventTerm(func=reset, mode='reset')


@configclass
class ContinuousEnvCfg(base.CycleEnvCfg):
    actions = ActionsCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    events = EventsCfg()
    terminations = DonesCfg()
    steering_design: str = 'direct'
    hold_range: tuple = (8., 12.)
    active_seconds: float = 24.
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 32.
