"""Body IMU orientation.

The ``imu`` site is fixed on ``lower_body`` with no extra rotation, so its axes
are the body axes: +X forward, +Y left, +Z up.
"""

import numpy as np


def read_rpy(model, data):
    """Roll, pitch, yaw in radians. Intrinsic ZYX."""
    rotation = np.array(data.site_xmat[model.site("imu").id], dtype=float).reshape(3, 3)
    pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])))
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return roll, pitch, yaw
