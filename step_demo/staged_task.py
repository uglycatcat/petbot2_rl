"""Restart14 objectives. Reward state is privileged; it never controls actions.

All running costs are bounded by MAX_RATE. At physical failure the worst
remaining running cost is charged, so termination cannot avoid STOP costs.
The auxiliary task is explicitly a separate known STOP command, not climbing.
"""
import torch

MAX_RATE = 6.0
FAILURE_COST = 50.0
SUCCESS_BONUS = 100.0


def advance_intent(previous, landed, four_top, drove):
    candidate = torch.where(landed, 1, 0)
    candidate = torch.where(landed & four_top, 2, candidate)
    candidate = torch.where(drove, 3, candidate)
    return torch.maximum(previous, candidate)


def costs(intent, wheel_request, hip_request, hip_position, hip_velocity,
          speed, yaw_rate, lateral, yaw, pitch, pitch_rate, repeated_flight):
    """Per-second costs: bounded but with nonzero slope over common stop errors."""
    stop = intent >= 3
    home = intent >= 2
    # No stop cost while climbing or doing the legally required short drive.
    wheel = 1.5 * (wheel_request / 25.).clamp(0., 1.) * stop
    motion = 1.5 * (speed / .75).clamp(0., 1.) * stop
    hip = .5 * ((hip_request.abs()/1.2).clamp_max(1.)
                + (hip_position.abs().amax(-1)/1.2).clamp_max(1.)) * home
    quiet = .25 * (hip_velocity.abs().amax(-1)/10.).clamp_max(1.) * home
    line = .25 * ((lateral.abs()/.15).clamp_max(1.)
                  + (yaw.abs()/.5).clamp_max(1.))
    turning = .25 * (yaw_rate.abs()/1.).clamp_max(1.) * stop
    # Secondary quality target throughout motion; not a hard limit on lifting.
    body = .1 * ((pitch.abs()/1.2).clamp_max(1.)
                 + (pitch_rate.abs()/10.).clamp_max(1.))
    flight = .5 * repeated_flight.float()
    components = torch.stack((wheel, motion, hip, quiet, line, turning, body, flight), -1)
    return components.sum(-1), components


def terminal_value(success, failed, elapsed, timeout):
    remaining = (timeout-elapsed).clamp_min(0.)
    return SUCCESS_BONUS*success.float() - failed.float()*(FAILURE_COST+MAX_RATE*remaining)


def body_pitch(quaternion, angular_velocity_world):
    # Isaac tensors here use xyzw. Euler pitch and its world-rate expression.
    x, y, z, w = quaternion.unbind(-1)
    pitch = torch.asin((2.*(w*y-z*x)).clamp(-1., 1.))
    yaw = torch.atan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z))
    rate = -yaw.sin()*angular_velocity_world[:, 0]+yaw.cos()*angular_velocity_world[:, 1]
    return pitch, rate


class FrontFlight:
    """20 ms debounced physical flight; contact gaps alone do not count.

    A quality violation stays latched but does not terminate physics early.
    Full success additionally requires exactly one legal flight per wheel.
    """
    def __init__(self, n, device):
        self.count = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.air = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.off_time = torch.zeros(n, 2, device=device)
        self.floor_time = torch.zeros_like(self.off_time)
        self.top_time = torch.zeros_like(self.off_time)
        self.returned_floor = torch.zeros_like(self.air)
        self.landed_top = torch.zeros_like(self.air)

    def reset(self, ids):
        for value in vars(self).values():
            value[ids] = 0

    def update(self, minimum, force, top_force, edge, height, dt):
        z, x = minimum[..., 2], minimum[..., 0]
        over = x > edge[:, None]+.0005
        plane = torch.where(over, height, 0.)
        magnitude = force.norm(dim=-1)
        off = (z-plane > .004) & (magnitude < .02)
        floor = (z <= .002) & (magnitude > .02)
        top = over & (z >= height-.002) & (z <= height+.004) & (top_force[..., 2] > .02)
        self.off_time = torch.where(off, self.off_time+dt, 0.)
        self.floor_time = torch.where(floor, self.floor_time+dt, 0.)
        self.top_time = torch.where(top, self.top_time+dt, 0.)
        leaving = ~self.air & (self.off_time >= .02-1e-7)
        self.count += leaving.long()
        self.air |= leaving
        on_floor = self.air & (self.floor_time >= .02-1e-7)
        on_top = self.air & (self.top_time >= .02-1e-7)
        self.returned_floor |= on_floor
        self.landed_top |= on_top
        self.air &= ~(on_floor | on_top)

    @property
    def invalid(self):
        return ((self.count > 1) | self.returned_floor).any(-1)

    @property
    def complete(self):
        return ((self.count == 1) & self.landed_top & ~self.air).all(-1) & ~self.invalid
