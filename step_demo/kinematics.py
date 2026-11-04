"""World-frame lower-body pitch and its physical time derivative."""
import torch


def body_pitch(quaternion, angular_velocity_world):
    # Isaac tensors here use xyzw. Euler pitch and its world-rate expression.
    x, y, z, w = quaternion.unbind(-1)
    pitch = torch.asin((2.*(w*y-z*x)).clamp(-1., 1.))
    yaw = torch.atan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z))
    rate = -yaw.sin()*angular_velocity_world[:, 0]+yaw.cos()*angular_velocity_world[:, 1]
    return pitch, rate
