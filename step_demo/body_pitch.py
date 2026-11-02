"""World-referenced lower_body pitch diagnostics and soft running costs.

Quaternions are xyzw. Body/hip joint angles are deliberately not inputs.
These functions do not modify observations, actions, physics or termination.
"""
import torch


def pitch_cost_scale_at(schedule, completed_updates):
    """Training curriculum only; evaluation always uses the full objective.

    Completed update counts select the next rollout's scale. No physical state
    or episode reset changes this global learning schedule.
    """
    if schedule is None:
        return 1.
    fraction = max(0., min(1., (completed_updates-schedule['start_update']) /
                           (schedule['end_update']-schedule['start_update'])))
    return schedule['initial_scale']+(1.-schedule['initial_scale'])*fraction


def _angles(q):
    x, y, z, w = q.unbind(-1)
    roll = torch.atan2(2*(w*x+y*z), 1-2*(x.square()+y.square()))
    pitch = torch.asin((2*(w*y-x*z)).clamp(-1., 1.))
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y.square()+z.square()))
    return roll, pitch, yaw


def world_pitch_state(quaternion, angular_velocity_world):
    """ZYX pitch and its derivative, using world angular velocity.

    A world-Y angular velocity alone is not the pitch derivative when yaw is
    nonzero. The original fall guard keeps normal operation away from gimbal lock.
    """
    _, pitch, yaw = _angles(quaternion)
    rate = -yaw.sin()*angular_velocity_world[..., 0]+yaw.cos()*angular_velocity_world[..., 1]
    return pitch, rate


def sensor_pitch_state(estimated_quaternion, gyro_body):
    """The corresponding proprioceptive estimate for the lower_body-mounted IMU."""
    roll, pitch, _ = _angles(estimated_quaternion)
    rate = roll.cos()*gyro_body[..., 1]-roll.sin()*gyro_body[..., 2]
    return pitch, rate


def pitch_costs(pitch, rate, dt):
    """Engineering weights, not hard/user-specified angle or rate limits.

    Charge fast rotation everywhere and only the excess angle beyond 0.35 rad.
    Integrate at physics cadence, including contact transients and rear ascent.
    No positive reward is earned for standing still or holding a tilted pose.
    """
    return dt*.1*rate.square(), dt*2.*(pitch.abs()-.35).clamp_min(0.).square()
