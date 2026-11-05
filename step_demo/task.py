"""One STEP task, advanced at the physics rate; no actuator overrides.

Simulation truth is confined to this task and the critic. The actor receives
only its sensor stream. A milestone is never undone by contact noise. Losing
support/pose while stopping resets the hold, not the completed drive.
"""
from dataclasses import dataclass
from enum import IntEnum

import torch

from .config import AcceptanceConfig, TerrainConfig


class Phase(IntEnum):
    APPROACH = 0
    FRONT_FLIGHT = 1
    REAR_ASCENT = 2
    ALIGN = 3
    DRIVE = 4
    STOP = 5
    COMPLETE = 6
    FAILED = 7


class Failure(IntEnum):
    NONE = 0
    FRONT_COLLISION = 1
    REAR_AIR = 2
    FALL = 3
    PATH = 4
    FRONT_FLIGHT = 5
    PLATFORM_EXIT = 6
    TIMEOUT = 7


@dataclass
class PhysicsSample:
    forward: torch.Tensor
    lateral: torch.Tensor
    yaw: torch.Tensor
    hip: torch.Tensor
    hip_velocity: torch.Tensor
    wheel_velocity: torch.Tensor
    speed: torch.Tensor
    yaw_rate: torch.Tensor
    pitch: torch.Tensor
    pitch_rate: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor
    clearance: torch.Tensor
    force: torch.Tensor
    platform_force: torch.Tensor
    contact_point: torch.Tensor
    body_invalid: torch.Tensor


class StepTask:
    """Explicit monotonic milestones and substep-quality state.

    Force-bearing top contact is classified by the actual contact position,
    not a wheel AABB penetration threshold. A short force reporting gap may
    preserve support only within 4 mm of terrain. Every force-bearing landing
    ends a flight immediately. A geometry-confirmed, force-free lift is counted
    in the same 2 ms sample; short second flights are not hidden by a timer.
    """
    support_grace_s = .02
    extra_hold_s = 2.
    quality_budget = 5.

    def __init__(self, n, device, acceptance: AcceptanceConfig,
                 terrain: TerrainConfig, edge_x, center_y, physics_dt=.002):
        self.a, self.t = acceptance, terrain
        self.edge_x, self.center_y = edge_x, center_y
        self.dt = physics_dt
        self.deadline_ticks = round(acceptance.timeout_s / physics_dt)
        if abs(self.deadline_ticks*physics_dt-acceptance.timeout_s) > 1e-7:
            raise ValueError('Deadline must be an integer number of physics steps')
        self.n, self.device = n, device
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros((n, *shape), device=device, dtype=dtype)
        self.phase = zeros(dtype=torch.long)
        self.failure = zeros(dtype=torch.long)
        self.ticks = zeros(dtype=torch.long)
        self.flight_count = zeros(2, dtype=torch.long)
        self.air = zeros(2, dtype=torch.bool)
        self.off_ticks = zeros(2, dtype=torch.long)
        self.landed_top = zeros(2, dtype=torch.bool)
        self.returned_floor = zeros(2, dtype=torch.bool)
        self.top_age = zeros(4) + 1.
        self.supported_top = zeros(4, dtype=torch.bool)
        self.rear_off_ticks = zeros(2, dtype=torch.long)
        self.drive_distance = zeros()
        self.drive_origin = zeros()
        self.drive_previous_x = zeros()
        self.drive_previous_ready = zeros(dtype=torch.bool)
        self.hold_ticks = zeros(dtype=torch.long)
        self.quality_cost = zeros()
        self.previous_potential = zeros()
        self.best_legal_potential = zeros()
        self.front_hit = zeros(dtype=torch.bool)
        self.ready = zeros(dtype=torch.bool)
        self.top_loaded = zeros(4, dtype=torch.bool)
        self._memory_names = tuple(k for k, v in vars(self).items()
                                   if isinstance(v, torch.Tensor) and k not in ('edge_x', 'center_y'))

    def reset(self, ids):
        for name in self._memory_names:
            getattr(self, name)[ids] = 1. if name == 'top_age' else 0

    @property
    def completed(self):
        return self.phase == Phase.COMPLETE

    @property
    def terminal(self):
        return self.completed | (self.failure != Failure.NONE)

    @property
    def hold_s(self):
        return self.hold_ticks * self.dt

    @property
    def elapsed_s(self):
        return self.ticks * self.dt

    def _fail(self, condition, reason, active):
        self.failure = torch.where(active & condition & (self.failure == 0),
                                   int(reason), self.failure)

    def _contacts(self, s, active):
        p, normal = s.contact_point, s.platform_force
        finite = torch.isfinite(p).all(-1)
        force_present = normal.norm(dim=-1) > .02
        point_on_top = (finite & (p[..., 0] > self.edge_x[:, None]+.0005)
                        & (p[..., 0] < self.edge_x[:, None]+self.t.platform_length_m)
                        & ((p[..., 1]-self.center_y[:, None]).abs() < self.t.platform_width_m/2)
                        & ((p[..., 2]-self.t.height_m).abs() <= .004))
        loaded = point_on_top & (normal[..., 2] > .02)
        # Missing/nonfinite contact positions cannot certify safe top loading.
        # Aggregated platform contact may include both top and riser normals.
        # A valid top point must not hide a simultaneous front-face collision.
        near_edge = s.minimum[:, :2, 0] <= self.edge_x[:, None]+.0005
        side_normal = normal[:, :2, :2].norm(dim=-1) > .01
        forbidden = (force_present[:, :2] & ~loaded[:, :2]) | (near_edge & side_normal)
        self.front_hit |= forbidden.any(-1) & active
        age = torch.where(loaded, 0., (self.top_age+self.dt).clamp_max(1.))
        self.top_age = torch.where(active[:, None], age, self.top_age)
        inside = ((s.minimum[..., 0] > self.edge_x[:, None]+.0005)
                  & (s.maximum[..., 0] < self.edge_x[:, None]+self.t.platform_length_m)
                  & ((s.minimum[..., 1]-self.center_y[:, None]).abs() < self.t.platform_width_m/2)
                  & ((s.maximum[..., 1]-self.center_y[:, None]).abs() < self.t.platform_width_m/2))
        # A loaded contact remains valid despite small collision-mesh penetration.
        support = (loaded | ((age <= self.support_grace_s+1e-7) & (s.clearance <= .004))) & inside
        self.supported_top = torch.where(active[:, None], support, self.supported_top)
        self.top_loaded = torch.where(active[:, None], loaded, self.top_loaded)
        return loaded

    def advance(self, s: PhysicsSample):
        """Consume one actual physics sample and return its reward increment.

        Real failure takes precedence over success. A valid hold completed at
        exactly the deadline succeeds; a later completion cannot be observed.
        Terminal rows remain latched until reset and cannot pay twice.
        """
        active = ~self.terminal
        self.ticks += active.long()
        old_phase = self.phase.clone()
        loaded = self._contacts(s, active)
        magnitude = s.force.norm(dim=-1)
        # Before the first top landing, a front wheel takes off from the FLOOR.
        # Distance to the nearby vertical face must not erase a real takeoff.
        # After top landing, a relift is measured above the platform surface.
        front_plane = self.landed_top.to(s.minimum.dtype)*self.t.height_m
        front_off = ((s.minimum[:, :2, 2]-front_plane > .004) & (magnitude[:, :2] < .02))
        rear_off = (s.clearance[:, 2:] > .004) & (magnitude[:, 2:] < .02)
        next_ticks = torch.where(front_off, self.off_ticks+1, 0)
        self.off_ticks = torch.where(active[:, None], next_ticks, self.off_ticks)
        leaving = ~self.air & front_off & active[:, None]
        self.flight_count += leaving.long()
        self.air |= leaving
        # Any genuine loaded landing ends this flight, including an 18 ms landing.
        floor = (s.minimum[:, :2, 2] <= .004) & (magnitude[:, :2] > .02) & ~loaded[:, :2]
        floor_landing = self.air & floor & active[:, None]
        top_landing = self.air & loaded[:, :2] & active[:, None]
        self.returned_floor |= floor_landing
        self.landed_top |= top_landing
        self.air &= ~(floor_landing | top_landing)
        rear_ticks = torch.where(rear_off, self.rear_off_ticks+1, 0)
        self.rear_off_ticks = torch.where(active[:, None], rear_ticks, self.rear_off_ticks)

        self._fail(self.front_hit, Failure.FRONT_COLLISION, active)
        self._fail(rear_off.any(-1), Failure.REAR_AIR, active)
        self._fail(s.body_invalid, Failure.FALL, active)
        self._fail((s.forward < -.05) | (s.lateral.abs() > .15), Failure.PATH, active)
        invalid_flight = ((self.flight_count > 1) | self.returned_floor
                          | (loaded[:, :2] & (self.flight_count == 0))).any(-1)
        self._fail(invalid_flight, Failure.FRONT_FLIGHT, active)
        beyond = (s.maximum[..., 0] >= self.edge_x[:, None]+self.t.platform_length_m).any(-1)
        self._fail(beyond, Failure.PLATFORM_EXIT, active)
        valid = active & (self.failure == 0)

        def milestone(condition, phase):
            self.phase = torch.where(valid & condition & (self.phase < phase), int(phase), self.phase)
        milestone((self.flight_count > 0).any(-1), Phase.FRONT_FLIGHT)
        milestone(self.landed_top.all(-1) & ~self.air.any(-1), Phase.REAR_ASCENT)
        milestone((self.phase >= Phase.REAR_ASCENT) & self.supported_top.all(-1), Phase.ALIGN)
        aligned = (s.lateral.abs() <= self.a.line_tolerance_m) & (s.yaw.abs() <= self.a.heading_tolerance_rad)
        home = s.hip.abs().amax(-1) <= self.a.hip_tolerance_rad
        ready = (self.landed_top.all(-1) & ~self.air.any(-1) & self.supported_top.all(-1)
                 & aligned & home & valid)
        self.ready = torch.where(active, ready, self.ready)
        milestone(self.ready, Phase.DRIVE)
        entering = valid & (old_phase < Phase.DRIVE) & (self.phase == Phase.DRIVE)
        self.drive_origin = torch.where(entering, s.forward, self.drive_origin)
        # Unqualified forward motion cannot pay. All backward motion still
        # reduces net displacement, so toggling ready cannot farm distance.
        unqualified = valid & (old_phase == Phase.DRIVE) & ~(self.ready & self.drive_previous_ready)
        excluded_forward = torch.where(unqualified, (s.forward-self.drive_previous_x).clamp_min(0.), 0.)
        self.drive_origin += excluded_forward
        distance = (s.forward-self.drive_origin).clamp(0., self.a.post_climb_distance_m)
        self.drive_distance = torch.where(valid & (self.phase >= Phase.DRIVE), distance, self.drive_distance)
        self.drive_previous_x = torch.where(active, s.forward, self.drive_previous_x)
        self.drive_previous_ready = torch.where(active, self.ready, self.drive_previous_ready)
        drove = self.drive_distance+1e-7 >= self.a.post_climb_distance_m
        milestone((self.phase == Phase.DRIVE) & self.ready & drove, Phase.STOP)
        stopped = ((self.phase == Phase.STOP) & self.ready & (s.speed <= self.a.stop_speed_m_s)
                   & drove & (s.yaw_rate.abs() <= self.a.stop_yaw_rate_rad_s))
        self.hold_ticks = torch.where(active, torch.where(stopped, self.hold_ticks+1, 0), self.hold_ticks)
        success = stopped & (self.hold_s+1e-7 >= self.a.stop_hold_s+self.extra_hold_s)
        milestone(success, Phase.COMPLETE)
        self._fail((self.ticks >= self.deadline_ticks) & ~self.completed, Failure.TIMEOUT, active)
        self.phase = torch.where(self.failure != 0, int(Phase.FAILED), self.phase)

        # Secondary bounded movement cost. Every valid success still dominates
        # every failure, independent of duration and intermediate shaping.
        rate = (.003*s.hip_velocity.square().mean(-1) + .2*s.pitch.square()
                + .02*s.pitch_rate.square() + .01*(s.wheel_velocity.abs()-25.).clamp_min(0.).square().mean(-1))
        new_cost = torch.minimum(self.quality_cost+rate.clamp_max(5.)*self.dt,
                                 torch.full_like(rate, self.quality_budget))
        cost_delta = torch.where(active, new_cost-self.quality_cost, 0.)
        self.quality_cost += cost_delta
        potential = self.potential(s)
        # Preserve bounded learning signal for LEGAL partial progress. Never
        # refresh it on an invalid/timeout sample, even if position increased.
        self.best_legal_potential = torch.where(active & ~self.terminal,
            torch.maximum(self.best_legal_potential, potential), self.best_legal_potential)
        # gamma=1, terminal potential=0: shaping itself telescopes to zero.
        # Failed terminal return retains at most 6 legitimate progress points.
        potential = torch.where(self.terminal, 0., potential)
        shaping = torch.where(active, potential-self.previous_potential, 0.)
        self.previous_potential = torch.where(active, potential, self.previous_potential)
        failure_return = -10.+.5*self.best_legal_potential.clamp(0.,12.)
        terminal_reward = 40.*(active & self.completed)+failure_return*(active & (self.failure != 0))
        return shaping-cost_delta+terminal_reward

    def potential(self, s):
        approach = 2.*(s.forward/.16).clamp(0., 1.)
        proximity = ((.08-(self.edge_x[:, None]-s.maximum[:, :2, 0]))/.08).clamp(0., 1.)
        lift = (s.minimum[:, :2, 2]/(self.t.height_m+.005)).clamp(0., 1.)
        # A near-edge vertical hop used to exhaust the same four-point band
        # as a real front landing. Keep a small lift signal, then pay for
        # horizontal crossing while both fronts clear the edge and the rear
        # wheels remain close to their supporting terrain. These quantities
        # shape reward only; contact/landing acceptance above is unchanged.
        rear_support = (1.-s.clearance[:, 2:]/.004).clamp(0., 1.).amin(-1)
        width = (s.maximum[:, :2, 0]-s.minimum[:, :2, 0]).clamp_min(.001)
        crossing = ((s.maximum[:, :2, 0]-self.edge_x[:, None])/width).clamp(0., 1.)
        clearance = ((s.minimum[:, :2, 2]-(self.t.height_m-.004))/.009).clamp(0., 1.)
        single_flight = ((self.flight_count == 1).all(-1)
                         & ~self.returned_floor.any(-1) & ~self.front_hit)
        lift_credit = .5*(proximity*lift).mean(-1)
        crossing_credit = 1.5*(crossing*clearance).amin(-1)*single_flight
        value = approach+rear_support*(lift_credit+crossing_credit)  # <= 4
        # Actual qualified front landing is worth at least one more point
        # than ANY pre-landing pose. Rear ascent then fills [5, 6], joining
        # the unchanged alignment/drive/STOP bands at 6/7/8.
        rear = 5.+(s.minimum[:, 2:, 2]/self.t.height_m).clamp(0., 1.).mean(-1)
        value = torch.where(self.phase >= Phase.REAR_ASCENT, rear, value)
        home = 6.+(1.-s.hip.abs().amax(-1)/.8).clamp(0., 1.)
        value = torch.where(self.phase >= Phase.ALIGN, home, value)
        value = torch.where(self.phase >= Phase.DRIVE, 7.+self.drive_distance/self.a.post_climb_distance_m, value)
        stop = 8.+2.*torch.exp(-s.speed/.1)+2.*(self.hold_s/(self.a.stop_hold_s+self.extra_hold_s)).clamp(0., 1.)
        stop_qualified = (self.phase >= Phase.STOP) & self.ready & (self.drive_distance+1e-7 >= self.a.post_climb_distance_m)
        return torch.where(stop_qualified, stop, value)

    def critic_memory(self):
        """Every persistent task variable, without lossy history clipping.

        Physical state is supplied by the adapter as a separate critic group.
        Online normalization is deliberately disabled in both networks.
        """
        values = []
        scales = {'ticks': self.deadline_ticks, 'off_ticks': 10., 'rear_off_ticks': 10.,
                  'hold_ticks': (self.a.stop_hold_s+self.extra_hold_s)/self.dt,
                  'quality_cost': self.quality_budget, 'previous_potential': 12.,
                  'best_legal_potential': 12., 'phase': 7.}
        for name in self._memory_names:
            values.append(getattr(self, name).float().reshape(self.n, -1)/scales.get(name, 1.))
        return torch.cat(values, -1)

    def snapshot(self, ids):
        return {name: getattr(self, name)[ids].tolist() for name in self._memory_names}
