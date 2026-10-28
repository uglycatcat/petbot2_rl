"""Four-wheel drive limits and control, independent of a learned policy envelope."""
import math

import torch

from .core.full_range import project_command
from .core.continuous import yaw_feedforward


class OrdinaryDrive:
    frame_input_rate = .5  # rad/s of operator target; Q/E release holds the target.
    frame_target_slew = 3.  # Independent ordinary handoff/initial-pose smoothing.

    def __init__(self, spec):
        wheels = [spec['drives'][name]['control_range'] for name in spec['velocity_joints']]
        self.wheel_limit = min(min(-lo, hi) for lo, hi in wheels)
        self.speed_limit = .03*self.wheel_limit
        self.yaw_limit = .06/.10*self.wheel_limit
        frames = [spec['joints'][name]['range'] for name in ('left_frame', 'right_frame')]
        self.frame_range = (max(x[0] for x in frames), min(x[1] for x in frames))
        self.default_frame = spec['joints']['left_frame']['default']
        assert self.default_frame == spec['joints']['right_frame']['default']
        self.integral = torch.zeros(1, 2)

    def validate(self, forward, yaw):
        if not all(math.isfinite(x) for x in (forward, yaw)):
            raise ValueError('Nonfinite operator command')
        if abs(forward)>self.speed_limit+1e-7 or abs(yaw)>self.yaw_limit+1e-7:
            raise ValueError('Operator command exceeds source actuator envelope')

    def wheels(self, command, motion, dt):
        command = project_command(command, self.wheel_limit)
        error = command-motion
        integral = (self.integral+dt*error*error.new_tensor([20., 8.])).clamp(
            min=error.new_tensor([-5., -2.]), max=error.new_tensor([5., 2.]))
        common = command[:, 0]/.03+10.*error[:, 0]+integral[:, 0]
        difference = yaw_feedforward(command[:, 1])+2.*error[:, 1]+integral[:, 1]
        desired = torch.stack((common-difference, common+difference), -1)
        # Conditional integration at the wheel ceiling; no policy command or
        # wheel slew is used here. Source velocity servos enforce torque limits.
        saturation = desired.abs().amax(-1)>self.wheel_limit
        self.integral[:] = torch.where(saturation[:, None], self.integral, integral)
        return desired.clamp(-self.wheel_limit, self.wheel_limit)
