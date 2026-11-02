"""Deployable task memory using the initial terrain packet and proprioception.

This estimates rear-wheel passage and a subsequent short drive. It is a policy
input, not a contact detector, success criterion or action override. Simulator
phase, contacts and ground-truth pose are deliberately absent from the interface.
"""
import torch


MEMORY_FIELDS = ('estimated_drive_started', 'estimated_drive_fraction', 'stop_intent')


class SensorStopMemory:
    def __init__(self, num_envs, device, drive_distance=.15, hip_tolerance=.05):
        self.drive_distance = drive_distance
        self.hip_tolerance = hip_tolerance
        self.home_time = torch.zeros(num_envs, device=device)
        self.anchor = torch.zeros_like(self.home_time)
        self.started = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.fraction = torch.zeros_like(self.home_time)
        self.stop_intent = torch.zeros_like(self.started)

    def reset(self, ids):
        for name in ('home_time', 'anchor', 'started', 'fraction', 'stop_intent'):
            getattr(self, name)[ids] = 0

    def update(self, packet, com_position, rear_position, hip_position, dt):
        # Coordinates are relative to the initial COM. The rear axle estimate
        # comes from encoder kinematics rotated by integrated IMU orientation.
        normal_x = packet[:, 2].sin()
        normal_y = -packet[:, 2].cos()
        rear_projection = rear_position[:, 0]*normal_x + rear_position[:, 1]*normal_y
        # Source rear radius is 3 cm; a further 1 cm margin covers the small
        # left/right projection difference in the intended straight approach.
        passed = rear_projection > packet[:, 1]+.04
        home = hip_position.abs().amax(-1) <= self.hip_tolerance
        self.home_time = torch.where(passed & home, self.home_time+dt, 0.)
        entering = ~self.started & (self.home_time >= .06-1e-7)
        self.anchor = torch.where(entering, com_position[:, 0], self.anchor)
        self.started |= entering
        progress = ((com_position[:, 0]-self.anchor)/self.drive_distance).clamp(0., 1.)
        self.fraction = torch.maximum(self.fraction, progress*self.started)
        self.stop_intent |= self.started & (self.fraction >= 1.)

    def observation(self):
        return torch.stack((self.started.float(), self.fraction, self.stop_intent.float()), dim=-1)


def wheel_request(action, mapping='affine'):
    """Nonnegative common speed; executed-target normalization stays unchanged."""
    if mapping == 'affine':
        return (action+1.)*12.5
    if mapping == 'zero_centered':
        return action.clamp_min(0.)*25.
    raise ValueError('Unknown wheel action mapping')
