"""Continuous remote commands and measured-motion feedback for the full cycle."""
import math
import torch
from .cycle import CycleProgress
from .moving import wrap
from .stability import StabilityMonitor, coarse_tracking_score


def yaw_feedforward(command):
    # Candidate operating-point offset motivated by the fixed-model drive probes.
    # The policy must correct it; this is not an identified inverse plant model.
    return command*(.1/.06) + .7*torch.tanh(command/.03)


def joint_tracking_score(error, tolerance):
    scores = 1./(1.+(error/tolerance).square())
    return .25*scores.sum(-1)+.5*scores.prod(-1)


def instantaneous_tolerance_score(error,tolerance):
    """Smooth occupancy surrogate with a steeper slope at the existing tolerance."""
    normalized=(error/tolerance).abs()
    scores=torch.sigmoid(4.*(1.-normalized))/torch.sigmoid(normalized.new_tensor(4.))
    return .25*scores.sum(-1)+.5*scores.prod(-1)


def bound_cycle_features(features):
    """Keep continuous holding observations bounded without altering task clocks."""
    features[:, 27:31].clamp_(-.3, .3)
    features[:, 37].clamp_(0., 1.)
    features[:, 48].clamp_(0., 1.)
    return features


class ContinuousProgress(CycleProgress):
    def __init__(self, *args, hold_range=(8., 12.), instant_tracking_weight=.3, tracking_objective='mixed', **kwargs):
        super().__init__(*args, **kwargs)
        self.hold_range = hold_range
        assert 0.<=instant_tracking_weight<=1.
        self.instant_tracking_weight=instant_tracking_weight
        assert tracking_objective in ('mixed','instant_tolerance','stability_first')
        self.tracking_objective=tracking_objective
        self.hold_duration = torch.zeros(self.n, device=self.device)
        self.previous_position = torch.zeros(self.n, 2, device=self.device)
        self.previous_heading = torch.zeros(self.n, device=self.device)
        self.motion = torch.zeros(self.n, 2, device=self.device)
        self.interval_motion = torch.zeros_like(self.motion)
        self.error_integral = torch.zeros_like(self.motion)
        self.prepared_step = -1
        self.stability_monitor=StabilityMonitor(self.n,self.device,self.dt)
        self.travel_limit=3.+self.speed_limit*(hold_range[1]+12.)

    def sample_commands(self, ids):
        # Euler cycle: all 30 directed transitions between the six command classes.
        order = torch.tensor([0,1,0,2,0,3,0,4,0,5,1,2,1,3,1,4,1,5,2,3,2,4,2,5,3,4,3,5,4,5],device=self.device)
        self.mode[ids] = order[(ids+self.command_count[ids]) % len(order)]
        super().sample_commands(ids)
        self.next_command[ids] = self.since[ids]+3.+self.uniform(ids, 2)

    def reset(self, ids, pos, heading):
        super().reset(ids, pos, heading)
        self.hold_duration[ids] = self.hold_range[0]+(self.hold_range[1]-self.hold_range[0])*self.uniform(ids, 11)
        self.previous_position[ids] = pos[ids, :2]
        self.previous_heading[ids] = heading[ids]
        self.motion[ids] = 0.
        self.interval_motion[ids] = 0.
        self.error_integral[ids] = 0.
        self.prepared_step = -1
        self.stability_monitor.reset(ids)

    def update_motion(self, s):
        turn = wrap(s['yaw']-self.previous_heading)
        middle = self.previous_heading+.5*turn
        displacement = s['pos'][:, :2]-self.previous_position
        forward = (displacement*torch.stack((middle.cos(), middle.sin()), -1)).sum(-1)/self.dt
        self.interval_motion[:] = torch.stack((forward, turn/self.dt), -1)
        self.motion += (1.-math.exp(-self.dt/.12))*(self.interval_motion-self.motion)
        self.previous_position.copy_(s['pos'][:, :2])
        self.previous_heading.copy_(s['yaw'])

    def update(self, s, fixed):
        # Admission uses measured motion as well as physical four-wheel contact.
        ready = s.copy()
        tolerance = self.raw.new_tensor([.01, .025])+.25*self.raw.abs()
        attained = ((self.motion-self.raw).abs()<tolerance).all(-1)
        ready['four'] = s['four'] & (self.active | attained)
        external = self.external_mode.clone()
        due = (self.phase==2) & ~external & (self.phase_time+self.dt>=self.hold_duration)
        self.requested_mode[due] = 0
        # Disable only the parent's old since-takeover automatic landing clock.
        self.external_mode[:] = True
        result = super().update(ready, fixed)
        self.external_mode.copy_(external)
        self.error_integral[result['entered_balance']] = 0.
        integrate = self.phase==2
        limit = self.error_integral.new_tensor([.2, .6])
        value = self.error_integral+self.dt*(self.command-self.motion)
        self.error_integral[integrate] = torch.maximum(torch.minimum(value, limit), -limit)[integrate]
        tolerance = self.raw.new_tensor([.005, .015])+.25*self.command.abs()
        filtered_score = joint_tracking_score(self.motion-self.command, tolerance)
        instant = torch.stack((s['forward'], s['heading_rate']), -1)
        instant_score = joint_tracking_score(instant-self.command, tolerance)
        mix=self.instant_tracking_weight
        result['track_reward'] = (self.phase==2)*result['support']*((1.-mix)*filtered_score+mix*instant_score)
        if self.tracking_objective=='instant_tolerance':
            result['track_reward']=(self.phase==2)*result['support']*instantaneous_tolerance_score(instant-self.command,tolerance)
        elif self.tracking_objective=='stability_first':
            result['track_reward']=(self.phase==2)*result['support']*coarse_tracking_score(self.motion,self.command)
            # Support earns balance credit without requiring negligible instantaneous sway.
            result['balance_reward']=(self.phase==2)*result['support'].float()
        instability=self.stability_monitor.update(self.phase,s,self.motion,self.command,
            geometry_mode=self.transition_contract=='linear_heading')
        if getattr(self,'full_range',False):
            # A dedicated call-exit detector handles early return in rise/balance.
            # Tracking error during braking is not itself a physical fall.
            instability[:,0]=False;instability[:,3]=False
        result['instability']=instability.clone()
        result['instability_dwell']=self.stability_monitor.dwell.clone()
        result['failure'] |= instability.any(-1)
        result['completed'] &= ~result['failure']
        result['qualified'] &= ~result['failure']
        result['motion_attained'] = attained
        result['motion_feedback'] = self.motion.clone()
        result['interval_motion'] = self.interval_motion.clone()
        result['effective_error'] = self.motion-self.command
        finite = torch.isfinite(torch.cat((s['pos'], s['vel'], s['q']), -1)).all(-1)
        reasons = [(~finite), s['pos'][:, 2]<.025, s['g'][:, 2]>.2,
                   s['vel'][:, :3].norm(dim=-1)>5, s['vel'][:, 3:].norm(dim=-1)>30,
                   result['warmup_failed'], (self.phase==1)&(self.phase_time>getattr(self,'rise_timeout',3.)),
                   (self.phase==3)&(self.phase_time>4.)&(self.transition_contract=='legacy'),
                   self.active & ((s['pos'][:, :2]-self.rise_position).norm(dim=-1)>self.travel_limit)]
        reasons.extend(instability.unbind(-1))
        result['failure_code'] = sum(flag.long()*(1<<i) for i, flag in enumerate(reasons))
        return result
