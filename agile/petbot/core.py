"""Tensor math shared by task and CPU tests. Quaternions use wxyz."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class SuccessThresholds:
    tilt_rad: float = math.radians(15)
    joint_error_rad: float = 0.15
    linear_speed: float = 0.05
    angular_speed: float = 0.15
    position_joint_speed: float = 0.15
    wheel_speed: float = 0.5
    height_error: float = 0.02
    contact_force: float = 0.1
    hold_s: float = 1.0


def as_torch(value):
    """Lab 3 wraps simulator arrays; already-Torch values need no conversion."""
    return value if isinstance(value, torch.Tensor) else value.torch


def uniform_quaternions(count, device="cpu", generator=None):
    # Normalized isotropic 4D Gaussian is uniform on S3, hence uniform on SO(3).
    q = torch.randn(count, 4, device=device, generator=generator)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def success_components(
    gravity, joint_error, lin_vel, ang_vel, pos_vel, wheel_vel, height, target_height, wheel_forces, cfg
):
    finite = torch.isfinite(
        torch.cat((gravity, joint_error, lin_vel, ang_vel, pos_vel, wheel_vel, height[:, None], wheel_forces), dim=-1)
    ).all(dim=-1)
    return {
        "finite": finite,
        "upright": -gravity[:, 2] >= math.cos(cfg.tilt_rad),
        "configuration": joint_error.abs().amax(dim=-1) <= cfg.joint_error_rad,
        "height": (height - target_height).abs() <= cfg.height_error,
        "support": (wheel_forces > cfg.contact_force).all(dim=-1),
        "still": (lin_vel.norm(dim=-1) <= cfg.linear_speed)
        & (ang_vel.norm(dim=-1) <= cfg.angular_speed)
        & (pos_vel.abs().amax(dim=-1) <= cfg.position_joint_speed)
        & (wheel_vel.abs().amax(dim=-1) <= cfg.wheel_speed),
    }


class HoldTracker:
    """Track consecutive successful steps, retaining whether an episode ever recovered."""

    def __init__(self, count, dt, hold_s, device):
        self.required = math.ceil(hold_s / dt)
        self.steps = torch.zeros(count, dtype=torch.long, device=device)
        self.ever = torch.zeros(count, dtype=torch.bool, device=device)
        self.first_time = torch.full((count,), float("nan"), device=device)

    def update(self, good, elapsed):
        self.steps = torch.where(good, self.steps + 1, 0)
        held = self.steps >= self.required
        first = held & ~self.ever
        self.first_time[first] = elapsed[first]
        self.ever |= held
        return held

    def reset(self, ids):
        self.steps[ids] = 0
        self.ever[ids] = False
        self.first_time[ids] = float("nan")


def sample_joints(count, names, spec, device, generator=None):
    # Continuous wheels have no finite position limits: angles are sampled in [-pi, pi].
    limits = [spec["joints"][name]["range"] or [-math.pi, math.pi] for name in names]
    bounds = torch.tensor(limits, device=device)
    u = torch.rand(count, len(names), device=device, generator=generator)
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])


def observation_vector(gravity, angular_velocity, joint_error, joint_velocity, previous_action):
    return torch.cat((angular_velocity * 0.2, gravity, joint_error, joint_velocity * 0.05, previous_action), dim=-1)
