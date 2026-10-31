"""Scenario sampling, observations and rewards; simulation truths stay out of policy inputs."""
import math
import torch
from .controller import Controller, Phase, Reason
from .curriculum import Curriculum
from .geometry import WheelGeometry, wrap
from .model import wheel_geometry

OBSERVATION_FIELDS = [('gravity_body', 3), ('gyro_body', 3), ('estimated_motion', 2),
                      ('position_joints', 5), ('joint_rates_scaled', 5), ('wheel_rates_scaled', 2),
                      ('control_reference', 2), ('transition_motion', 2), ('executed_targets_scaled', 3),
                      ('phase_one_hot', 6), ('transition_heading_sin_cos', 2), ('phase_age_scaled', 1),
                      ('estimated_front_clearance', 2), ('ordinary_hip_target', 1)]
OBSERVATION_DIM = sum(size for _, size in OBSERVATION_FIELDS)

class Runtime:
    def __init__(self, n, device, dt, cfg, model):
        self.cfg, self.n, self.dt = cfg, n, dt
        self.geometry = WheelGeometry(wheel_geometry(), device)
        self.control = Controller(n, device, dt, cfg, model, self.geometry.track)
        self.course = Curriculum(cfg)
        self.stage_ranges = torch.tensor([[0., 0., 0., 0.], [0., 0., 0., 0.]]+cfg.motion_stages(), device=device)
        # Leave at least one second after the largest commanded reversal's reference ramp.
        self.stage_dwell = torch.full((len(self.stage_ranges),), 2., device=device)
        for axis, rate in enumerate((cfg.balance_linear_accel_m_s2, cfg.balance_yaw_accel_rad_s2)):
            if rate > 0:
                self.stage_dwell = torch.maximum(self.stage_dwell, 2*self.stage_ranges[:, axis]/rate+1.)
        self.generator = torch.Generator(device=device).manual_seed(cfg.seed)
        z = lambda: torch.zeros(n, device=device)
        self.level = z().long(); self.episode_age = z(); self.command_age = z()
        self.true_hold = z(); self.longest_hold = z(); self.peak_rise = z(); self.peak_land = z()
        self.balance_elapsed = z(); self.balance_seen = z().bool(); self.invoked = z().bool()
        self.episode_goal = z().bool(); self.true_cycle = z().bool(); self.failure = z().bool()
        self.balance_velocity_error = z()
        self.balance_duration = torch.full((n,), cfg.balance_train_s, device=device)
        self.short_cycle = z().bool()
        self.initial_hip = z(); self.takeover_motion = torch.zeros(n, 2, device=device)
        self.probe_level = None
        self.probe_duration = None
        self.probe_moving = True
        self.last = None

    def uniform(self, count, width=1):
        return torch.rand((count, width), device=self.level.device, generator=self.generator)

    def training_command(self, ids):
        ranges = self.stage_ranges[self.level[ids], :2]
        normalized = 2*self.uniform(len(ids), 2)-1
        category = (self.uniform(len(ids))[:, 0]*5).long()
        # Keep the continuous distribution and original categories. Mix in exact
        # boundaries after the low-range course, with independent symmetric signs.
        # No fixed evaluation sequence, future command, or teacher is supplied.
        eligible = (self.level[ids] >= 3).nonzero().flatten()
        if self.cfg.boundary_command_fraction and len(eligible):
            selected = self.uniform(len(eligible))[:, 0] < self.cfg.boundary_command_fraction
            edge_ids = eligible[selected]
            normalized[edge_ids] = torch.where(normalized[edge_ids] >= 0, 1., -1.)
        command = normalized*ranges
        command[category == 0] = 0  # stop
        command[category == 1, 1] = 0  # straight
        command[category == 2, 0] = 0  # pure turn
        return command

    def reset(self, ids):
        self.control.reset(ids)
        for name in ('episode_age', 'command_age', 'true_hold', 'longest_hold', 'peak_rise',
                     'peak_land', 'balance_elapsed', 'balance_seen', 'invoked', 'episode_goal',
                     'true_cycle', 'failure', 'initial_hip', 'takeover_motion', 'balance_velocity_error', 'short_cycle'):
            getattr(self, name)[ids] = 0
        level = self.course.level if self.probe_level is None else self.probe_level
        self.level[ids] = level
        if level >= 2 and self.probe_level is None:
            baseline = self.uniform(len(ids))[:, 0] < self.cfg.baseline_fraction
            # Preserve static cycles and earlier moving stages instead of dropping acquired tasks.
            if level == 2:
                self.level[ids[baseline]] = 1
            else:
                earlier = 1+(self.uniform(len(ids))[:, 0]*(level-1)).long()
                self.level[ids[baseline]] = earlier[baseline]
        moving = self.level[ids] >= 2
        ranges = self.stage_ranges[self.level[ids]]
        if self.probe_level is None:
            velocity = (2*self.uniform(len(ids))[:, 0]-1)*ranges[:, 2]
            hip = (2*self.uniform(len(ids))[:, 0]-1)*ranges[:, 3]
        else:
            # Deterministic paired initial states, independent of training random draws.
            velocity = (((ids//2) % 3).float()-1)*ranges[:, 2]
            hip = (((ids//6) % 3).float()-1)*ranges[:, 3]
            if not self.probe_moving:
                velocity.zero_(); hip.zero_()
        self.initial_hip[ids] = torch.where(moving, hip, 0.)
        command = torch.stack((torch.where(moving, velocity, 0.), torch.zeros_like(velocity)), -1)
        self.control.set_command(command, self.initial_hip[ids], ids)
        # Sample only training episode timing. The actor never sees the future Space time.
        duration = self.cfg.balance_train_s if self.probe_duration is None else self.probe_duration
        self.balance_duration[ids] = torch.where(self.level[ids] == 1, self.cfg.hold_s, duration)
        if self.probe_level is None and self.cfg.short_cycle_fraction > 0:
            moving_ids = ids[moving]
            short = self.uniform(len(moving_ids))[:, 0] < self.cfg.short_cycle_fraction
            sampled = self.cfg.short_balance_min_s + (self.cfg.short_balance_max_s-self.cfg.short_balance_min_s)*self.uniform(len(moving_ids))[:, 0]
            self.balance_duration[moving_ids] = torch.where(short, sampled, self.cfg.balance_train_s)
            self.short_cycle[moving_ids] = short

    def before_action(self, s):
        c = self.control
        ready = (c.phase == Phase.ORDINARY) & ~self.invoked & ~self.failure & (self.episode_age >= self.cfg.warmup_s)
        ready &= (s['gap'].abs().amax(-1) < self.cfg.ground_gap_m) & (s['rear_skew'].abs() < .02) & (s['g'][:, 2] < -.25)
        ids = ready.nonzero().flatten()
        if len(ids):
            c.request_rise(s, ids)
            self.takeover_motion[ids] = s['motion'][ids]
            self.invoked[ids] = True
        balance = c.phase == Phase.BALANCE
        self.command_age += balance*self.dt
        change = balance & (self.level >= 2) & (self.command_age >= self.stage_dwell[self.level])
        ids = change.nonzero().flatten()
        if len(ids):
            if self.probe_level is None:
                command = self.training_command(ids)
            else:
                # Intermediate speeds, reversal, stop, pure turns and paired arcs.
                table = s['motion'].new_tensor([[.5, 0], [-1, 0], [0, 0], [0, 1], [1, 1], [.5, -1], [-.5, 0], [0, 0]])
                if self.probe_level >= 3:
                    table = s['motion'].new_tensor([[.5, 0], [1, 0], [-1, 0], [-.5, 0], [0, 0],
                                                   [0, 1], [1, 1], [.5, -1], [0, 0]])
                index = (self.balance_elapsed[ids]/self.stage_dwell[self.level[ids]]).long() % len(table)
                command = table[index]*self.stage_ranges[self.level[ids], :2]
                command[:, 1] *= torch.where(ids % 2 == 0, 1., -1.)
            c.set_command(command, ids=ids); self.command_age[ids] = 0
        landing = balance & (self.level > 0) & (self.balance_elapsed >= self.balance_duration)
        c.request_land(s, landing.nonzero().flatten())
        c.update_motion_reference()
        return ready.nonzero().flatten()

    def advance(self, s, truth):
        c = self.control
        self.episode_age += self.dt
        result = c.update(s)
        before = result['phase_before']
        # Geometric support is dense enough for learning; actual force occupancy is reported separately.
        rear_near = (truth['rear_z'].amax(-1) < .046) & (truth['rear_z'].amin(-1) > .014)
        physical_support = rear_near & (truth['front_z'].amin(-1) > .12) & ~truth['body_contact']
        physical_support &= (s['g'][:, 2] < -.25) & (s['g'][:, 1].abs() < .35)
        in_balance = c.phase == Phase.BALANCE
        self.balance_seen |= in_balance & physical_support
        self.balance_elapsed += in_balance*self.dt
        self.true_hold.copy_(torch.where(in_balance & physical_support, self.true_hold+self.dt, 0.))
        self.longest_hold.copy_(torch.maximum(self.longest_hold, self.true_hold))
        stage_goal = (self.level == 0) & (self.true_hold >= self.cfg.hold_s)
        warmup_failed = ~self.invoked & (self.episode_age >= self.cfg.warmup_timeout_s)
        physical_fall = self.invoked & ((truth['root_height'] < .025) | truth['body_contact'])
        # Contact diagnostics may terminate training but never masquerade as deployable perception.
        fail = result['failure'] | warmup_failed | physical_fall
        c.reason[warmup_failed] = Reason.WARMUP_TIMEOUT
        c.reason[physical_fall & ~result['failure']] = Reason.FALL
        c.phase[fail] = Phase.EXIT
        self.failure |= fail
        self.episode_goal |= stage_goal & ~self.failure
        # A complete cycle requires a real sustained two-wheel interval and real four-wheel landing.
        required_hold = self.balance_duration
        real_ground = (truth['front_z'].amax(-1) < .065) & rear_near & ~truth['body_contact']
        valid_cycle = c.completed & self.balance_seen & (self.longest_hold >= required_hold-.15) & real_ground & ~self.failure
        self.true_cycle |= valid_cycle
        # Geometric handoff can occur without truth support: report failure, never a false success.
        mismatch = c.completed & ~valid_cycle
        c.reason[mismatch & ~self.failure] = Reason.SUPPORT_MISMATCH
        c.phase[mismatch] = Phase.EXIT
        self.failure |= mismatch
        result['failure'] = self.failure.clone()
        result['support_truth'] = physical_support
        result['goal'] = self.episode_goal.clone()
        result['complete'] = self.true_cycle.clone()
        result['done'] = self.failure | self.episode_goal | c.completed
        result['balance_required_s'] = self.balance_duration.clone()
        result['short_cycle'] = self.short_cycle.clone()
        result['perception_mismatch'] = result['support_estimate'] != physical_support
        result['reason'] = c.reason.clone()
        result['reward'] = self.reward(s, truth, result, rear_near)
        self.last = result
        return result

    @staticmethod
    def huber(value):
        absolute = value.abs()
        return torch.where(absolute <= 1., .5*value.square(), absolute-.5)

    def velocity_costs(self, s, balancing, landing):
        # Smooth the signed error (not its magnitude): distinguish persistent bias
        # from short physical fluctuations. It is reward state, never a policy input.
        error = s['motion'][:, 0]-self.control.motion_command()[:, 0]
        alpha = 1.-math.exp(-self.dt/self.cfg.balance_velocity_error_tau_s)
        self.balance_velocity_error.copy_(torch.where(balancing,
            self.balance_velocity_error+alpha*(error-self.balance_velocity_error), 0.))
        balance_cost = self.dt*self.cfg.balance_velocity_cost*self.huber(
            self.balance_velocity_error/self.cfg.balance_velocity_error_scale_m_s)*balancing
        # Preserve the speed measured at LAND entry, not zero or the last remote setpoint.
        landing_error = s['motion'][:, 0]-self.control.transition_motion[:, 0]
        land_cost = self.dt*self.cfg.landing_velocity_cost*self.huber(
            landing_error/self.cfg.landing_velocity_error_scale_m_s)*landing
        return balance_cost, land_cost

    def reward(self, s, truth, m, rear_near):
        c = self.control; before = m['phase_before']
        safe = ~m['failure']; rising = (before == Phase.RISE) & safe
        balancing = (before == Phase.BALANCE) & safe & m['support_truth']
        landing = (before == Phase.LAND) & safe
        lift = ((truth['front_z'].amin(-1)-.045)/.075).clamp(0, 1)*rear_near*~truth['body_contact']
        prepare = .25*(-s['g'][:, 0]/.8).clamp(0, 1)*rear_near
        phi = (prepare+lift)*rising
        gain = (phi-self.peak_rise).clamp_min(0)*rising
        self.peak_rise.copy_(torch.maximum(self.peak_rise, phi))
        land_phi = ((.15-truth['front_z'].amax(-1))/.105).clamp(0, 1)*rear_near*landing
        land_gain = (land_phi-self.peak_land).clamp_min(0)*landing
        self.peak_land.copy_(torch.maximum(self.peak_land, land_phi))
        error = s['motion']-c.motion_command()
        tracking = .5*torch.exp(-(error[:, 0]/.10).square())+.5*torch.exp(-(error[:, 1]/.30).square())
        transition = rising | landing
        heading_error = wrap(s['yaw']-c.yaw_reference)
        target_change = (c.executed-c.previous)/s['q'].new_tensor([2.2, 25., 25.])
        adjustment = (s['motion'][:, 0]-self.takeover_motion[:, 0]).abs()
        reward = 3*gain + 5*land_gain
        # Restore the early scratch task's dense bridge. Real partial lift and
        # rear support earn credit before the strict BALANCE entry event. A
        # grounded preparatory pitch alone remains net-negative and expires.
        prep_decay = (1-c.age/self.cfg.rise_prepare_decay_s).clamp(0, 1)
        support_duration = (c.support_run/self.cfg.support_enter_s).clamp(0, 1)
        bridge = rising*(self.cfg.rise_prepare_weight*prepare*prep_decay
                          + self.cfg.rise_partial_lift_weight*lift.square()
                          + self.cfg.rise_support_weight*m['support_truth']*(.2+.8*support_duration))
        reward += self.dt*bridge
        reward += 5*m['entered_balance']*m['support_truth']*safe
        reward += 10*m['goal']*safe + 30*m['complete']*safe
        reward += self.dt*((2+2*tracking)*balancing - .1*transition
                           - .15*transition*heading_error.square()
                           - .02*rising*adjustment - .002*target_change.square().sum(-1)*self.invoked)
        # The first unsuccessful lift still ends this invocation immediately.
        # A large early terminal penalty was cheaper to postpone until timeout,
        # suppressing exploration just below the lift detector. Grade only an
        # upright, rear-supported front return before any BALANCE entry; never
        # soften actual falls or loss of an established two-wheel state.
        early_return = (m['failure'] & (before == Phase.RISE)
                        & (m['reason'] == Reason.FRONT_RETURN) & ~c.entered
                        & rear_near & ~truth['body_contact'] & (truth['root_height'] >= .025)
                        & (s['g'][:, 2] < -.25) & (s['g'][:, 1].abs() < .35))
        penalty = torch.where(early_return, self.cfg.rise_front_return_penalty, 5.)*m['failure']
        balance_cost, land_cost = self.velocity_costs(s, balancing, landing)
        reward -= penalty+balance_cost+land_cost
        m['balance_velocity_error_mean'] = self.balance_velocity_error.clone()
        m['balance_velocity_cost'] = balance_cost
        m['landing_velocity_cost'] = land_cost
        m['failure_penalty'] = penalty
        m['early_front_return'] = early_return
        m['rise_bridge_reward'] = self.dt*bridge
        return reward

    def observation_stage_age(self):
        c = self.control
        if self.cfg.balance_observation_age_cap_s > 0:
            return torch.where(c.phase == Phase.BALANCE,
                               c.age.clamp_max(self.cfg.balance_observation_age_cap_s), c.age)
        return c.age

    def observation(self, s):
        c = self.control
        # All motion inputs are in the local/body frame; no world XY or absolute yaw.
        g, gyro, q = s['g'], s['gyro'], s['q']
        if self.probe_level is None:
            if self.cfg.imu_noise_std:
                g = g+torch.randn(g.shape, device=g.device, generator=self.generator)*self.cfg.imu_noise_std
                gyro = gyro+torch.randn(gyro.shape, device=g.device, generator=self.generator)*self.cfg.imu_noise_std
            if self.cfg.joint_noise_std:
                q = q+torch.randn(q.shape, device=q.device, generator=self.generator)*self.cfg.joint_noise_std
        yaw = wrap(c.yaw_reference-s['yaw'])
        yaw = torch.where((c.phase == Phase.RISE) | (c.phase == Phase.LAND), yaw, 0.)
        executed = c.executed/s['q'].new_tensor([1.2, 25., 25.])
        return torch.cat((g, gyro, s['motion'], q, s['qd']*.1, s['wheel']*.05,
                          c.motion_command(), c.transition_motion, executed,
                          torch.nn.functional.one_hot(c.phase, 6).float(),
                          torch.stack((yaw.sin(), yaw.cos()), -1), self.observation_stage_age()[:, None]/12,
                          s['gap'], c.ordinary_hip[:, None]), -1)
