"""Two normalized policy outputs become exactly symmetric actuator targets."""

from dataclasses import dataclass
import math

from .config import ControlConfig
from .model import ModelLimits


@dataclass(frozen=True)
class ActuatorTargets:
    hip_positions_rad: tuple[float, float]
    rear_velocities_rad_s: tuple[float, float]
    head_and_ears_rad: tuple[float, float, float]


def map_action(action: tuple[float, float], limits: ModelLimits,
               control: ControlConfig) -> ActuatorTargets:
    """hip [-1, 0, 1] -> [min, default, max]; wheel [-1, 1] -> [0, max].

    This maps targets only. It does not implement a position servo, smoothing,
    terrain script, balance controller or simulator drive.
    """
    if len(action) != 2 or not all(math.isfinite(a) and -1 <= a <= 1 for a in action):
        raise ValueError("Expected two finite normalized actions in [-1, 1]")
    hip, wheel = action
    home = control.default_hip_rad
    if not limits.hip_min_rad <= home <= limits.hip_max_rad:
        raise ValueError("Default hip target outside source limits")
    target = home + hip * ((limits.hip_max_rad - home) if hip >= 0
                           else (home - limits.hip_min_rad))
    speed = (wheel + 1) * .5 * limits.rear_speed_max_rad_s
    return ActuatorTargets((target, target), (speed, speed),
                           (control.head_rad, control.left_ear_rad, control.right_ear_rad))
