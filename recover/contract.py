"""One observation/action/success contract shared by training and MuJoCo validation."""
import math
import torch

OBS_DIM = 34
ACTION_DIM = 7
POSE_COUNT = 4
POSITION_LOW = (-1.3, -1., -1., -1., -1.)
POSITION_HIGH = (1.3, 3., 3., 1.2, 1.2)
GAMMA = 0.998
CONTROL_DT = 0.02
HOLD_S = 0.5
TILT_COS = math.cos(math.radians(25))


def quat_rotate_inverse(q, v):
    qv = q[:, 1:]
    return v * (2 * q[:, :1].square() - 1) - 2*q[:, :1]*torch.cross(qv, v, dim=-1) + 2*qv*(qv*v).sum(-1, keepdim=True)


def observation(root, q, qd, wheel, last_action, contacts, remaining):
    down = torch.zeros_like(root[:, :3]); down[:, 2] = -1
    gravity = quat_rotate_inverse(root[:, 3:7], down)
    gyro = quat_rotate_inverse(root[:, 3:7], root[:, 10:13])
    velocity = quat_rotate_inverse(root[:, 3:7], root[:, 7:10])
    low = torch.as_tensor(POSITION_LOW, device=q.device)
    high = torch.as_tensor(POSITION_HIGH, device=q.device)
    q_scaled = 2*(q-low)/(high-low)-1
    obs = torch.cat((gravity, gyro*.2, q_scaled, qd*.1, wheel/25,
                     last_action, velocity, contacts.float(), root[:, 2:3]*5,
                     remaining[:, None]), -1)
    return obs.clamp(-10, 10)


def posture(root, wheel_contact, other_force, wheel_velocity):
    # Body z dot world z for a wxyz quaternion; independent of heading.
    up = 1-2*(root[:, 4].square()+root[:, 5].square())
    upright = up > TILT_COS
    four = wheel_contact.all(-1)
    slow = ((root[:, 7:10].norm(dim=-1) < .15) &
            (root[:, 10:13].norm(dim=-1) < .4) &
            (wheel_velocity.abs().amax(-1) < 1.5))
    stable = upright & four & slow & (other_force < .5)
    potential = (up+1) + wheel_contact.float().mean(-1)
    return up, four, stable, potential
