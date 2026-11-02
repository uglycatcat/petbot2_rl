"""Exact inference observation methods extracted from round20 Runtime."""
import torch
from .geometry import wrap
from .controller import Phase

class Observation:
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
