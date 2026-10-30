"""Sensor-space geometry. Quaternion convention is Isaac Lab 3's xyzw.

Clearance is relative to the rear-wheel support plane, not an absolute height
sensor. It is an estimate: contact/height truth must independently audit it.
"""
import torch


def wrap(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def orientation(quaternion, linear_world, angular_world):
    x, y, z, w = quaternion.unbind(-1)
    heading = torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z)), -1)
    heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    gravity = torch.stack((2*(w*y-x*z), -2*(w*x+y*z), 2*(x*x+y*y)-1), -1)
    # Inverse quaternion rotation, independent of the global heading.
    vector = quaternion[:, :3]
    def inverse(v):
        t = 2*torch.cross(vector, v, dim=-1)
        return v - w[:, None]*t + torch.cross(vector, t, dim=-1)
    axis = torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)), -1)
    derivative = torch.cross(angular_world, axis, dim=-1)
    yaw_rate = (axis[:, 0]*derivative[:, 1]-axis[:, 1]*derivative[:, 0]) / axis[:, :2].square().sum(-1).clamp_min(.05)
    motion = torch.stack(((linear_world[:, :2]*heading).sum(-1), yaw_rate), -1)
    return gravity, torch.atan2(heading[:, 1], heading[:, 0]), inverse(angular_world), motion


class WheelGeometry:
    def __init__(self, geometry, device):
        self.geometry = {k: torch.tensor(v, device=device) for k, v in geometry.items()}
        self.track = abs(geometry['left_hip'][1]+geometry['left_rear'][1]
                         -geometry['right_hip'][1]-geometry['right_rear'][1])

    def estimate(self, gravity, hip):
        heights, defaults = [], []
        for side_index, side in enumerate(('left', 'right')):
            c, s = hip[:, side_index].cos(), hip[:, side_index].sin()
            for end in ('front', 'rear'):
                h, p = self.geometry[f'{side}_hip'], self.geometry[f'{side}_{end}']
                q = torch.stack((c*p[0]+s*p[2]+h[0], torch.ones_like(c)*(p[1]+h[1]),
                                 -s*p[0]+c*p[2]+h[2]), -1)
                heights.append((-gravity*q).sum(-1))
                defaults.append(h[2]+p[2])
        # left-front, left-rear, right-front, right-rear; remove nominal wheel radii.
        relative = torch.stack(heights, -1) - torch.stack(defaults)
        rear = relative[:, [1, 3]]
        gap = relative[:, [0, 2]] - rear.mean(-1, keepdim=True)
        return gap, rear[:, 0]-rear[:, 1]
