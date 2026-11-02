"""Privileged physical-substep single-flight contract, never an actor input."""
import torch


class FrontMotion:
    def __init__(self, num_envs, device):
        self.air = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.count = torch.zeros(num_envs, 2, dtype=torch.long, device=device)
        self.off_time = torch.zeros(num_envs, 2, device=device)
        self.floor_time = torch.zeros_like(self.off_time)
        self.top_time = torch.zeros_like(self.off_time)
        self.returned_floor = torch.zeros_like(self.air)
        self.landed_top = torch.zeros_like(self.air)

    def reset(self, ids):
        for name in ('air', 'count', 'off_time', 'floor_time', 'top_time',
                     'returned_floor', 'landed_top'):
            getattr(self, name)[ids] = 0

    def update(self, minimum, force, platform_normal, edge_x, height, dt):
        z, x = minimum[..., 2], minimum[..., 0]
        over_top = x > edge_x[:, None]+.0005
        plane = torch.where(over_top, height, 0.)
        magnitude = force.norm(dim=-1)
        off = (z-plane > .004) & (magnitude < .02)
        floor = (z <= .002) & (magnitude > .02)
        top = over_top & (z >= height-.002) & (z <= height+.004) & (platform_normal[..., 2] > .02)
        self.off_time = torch.where(off, self.off_time+dt, 0.)
        self.floor_time = torch.where(floor, self.floor_time+dt, 0.)
        self.top_time = torch.where(top, self.top_time+dt, 0.)
        leaving = ~self.air & (self.off_time >= .02-1e-7)
        self.count += leaving.long()
        self.air |= leaving
        floor_landing = self.air & (self.floor_time >= .02-1e-7)
        top_landing = self.air & (self.top_time >= .02-1e-7)
        self.returned_floor |= floor_landing
        self.landed_top |= top_landing
        self.air &= ~(floor_landing | top_landing)

    @property
    def invalid(self):
        return (self.returned_floor | (self.count > 1)).any(-1)

    @property
    def complete(self):
        return ((self.count == 1) & self.landed_top & ~self.air).all(-1) & ~self.invalid
