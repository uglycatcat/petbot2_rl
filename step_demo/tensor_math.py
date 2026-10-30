"""Torch-only math. Quaternion convention is xyzw (Isaac Lab 3)."""
import torch


def quat_mul(a, b):
    xyz = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + torch.cross(a[..., :3], b[..., :3], dim=-1)
    w = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, w), -1)


def rotate(q, v):
    uv = torch.cross(q[..., :3], v, dim=-1)
    return v + 2 * (q[..., 3:] * uv + torch.cross(q[..., :3], uv, dim=-1))


def inverse_rotate(q, v):
    return rotate(torch.cat((-q[..., :3], q[..., 3:]), -1), v)


def rotation_step(omega, dt):
    angle = omega.norm(dim=-1, keepdim=True) * dt
    return torch.cat((omega * (0.5 * dt * torch.sinc(angle / (2 * torch.pi))), torch.cos(angle / 2)), -1)


def heading(q):
    return torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                       1 - 2 * (q[:, 1].square() + q[:, 2].square()))
