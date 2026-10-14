"""IMU orientation and the upright / fallen flag.

The IMU site is fixed to ``lower_body`` with no extra rotation, so its axes are
the body axes: +X forward, +Y left, +Z up. Tilt is the angle between IMU +Z
and world +Z. Yaw alone does not count as a fall.
"""

import numpy as np

# Enter at 45° and clear only after returning below 35°, so the flag does not
# flicker while the body is held near the boundary.
FALLEN_ENTER_DEG = 45.0
FALLEN_EXIT_DEG = 35.0
FALLEN_ENTER_RAD = np.deg2rad(FALLEN_ENTER_DEG)
FALLEN_EXIT_RAD = np.deg2rad(FALLEN_EXIT_DEG)


def rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw in radians. Intrinsic ZYX, the same order as ROS RPY."""
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2]))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def tilt_from_world_up(rotation: np.ndarray) -> float:
    """Angle between IMU +Z and world +Z, in radians, range [0, pi]."""
    return float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))


def read_imu(model, data) -> tuple[np.ndarray, np.ndarray, float]:
    """Copy the IMU site rotation and return ``(rotation, rpy, tilt)``."""
    rotation = np.array(data.site_xmat[model.site("imu").id], dtype=float).reshape(3, 3)
    return rotation, rotation_to_rpy(rotation), tilt_from_world_up(rotation)


class FallJudge:
    def __init__(self):
        self.fallen = False

    def update(self, tilt: float) -> bool:
        if self.fallen:
            if tilt <= FALLEN_EXIT_RAD:
                self.fallen = False
        elif tilt >= FALLEN_ENTER_RAD:
            self.fallen = True
        return self.fallen
