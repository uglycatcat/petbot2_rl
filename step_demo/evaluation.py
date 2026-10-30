"""Privileged acceptance tracker, independent from actor observations/rewards.

Evidence comes from a future physics adapter. Synthetic tests only verify
sequencing; they do not establish contact detection or climbing ability.
"""

from dataclasses import dataclass
from enum import Enum
import math

from .config import AcceptanceConfig


@dataclass(frozen=True)
class EvaluationSample:
    dt_s: float
    forward_m: float
    lateral_m: float
    heading_rad: float
    speed_m_s: float  # magnitude of actual whole-body COM linear velocity
    yaw_rate_rad_s: float
    hip_error_rad: tuple[float, float]
    front_clearance_verified: tuple[bool, bool]  # swept clearance before top landing
    four_wheels_on_top: bool  # actual contact plus geometry, not chassis position alone
    front_face_or_edge_hit: bool  # any physical-substep forbidden contact
    rear_support_lost: bool  # geometry/contact evidence; needs physical validation
    body_collision_or_fall: bool
    deliberate_reverse: bool  # target violation, separate from measured rollback


class Stage(str, Enum):
    CROSSING = "crossing"
    PLATFORM_DRIVE = "platform_drive"
    STOPPING = "stopping"
    COMPLETE = "complete"
    FAILED = "failed"


class CompletionTracker:
    def __init__(self, config: AcceptanceConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.stage = Stage.CROSSING
        self.reason: str | None = None
        self.elapsed_s = 0.
        self.stopped_s = 0.
        self.drive_start_m: float | None = None
        self.front_clearance = [False, False]

    def update(self, s: EvaluationSample) -> Stage:
        numbers = (s.dt_s, s.forward_m, s.lateral_m, s.heading_rad, s.speed_m_s,
                   s.yaw_rate_rad_s, *s.hip_error_rad)
        if not all(math.isfinite(x) for x in numbers) or s.dt_s <= 0 or s.speed_m_s < 0:
            raise ValueError("Invalid evaluation sample")
        if len(s.hip_error_rad) != 2 or len(s.front_clearance_verified) != 2:
            raise ValueError("Both sides must be evaluated")
        if self.stage in (Stage.COMPLETE, Stage.FAILED):
            return self.stage
        self.elapsed_s += s.dt_s
        for failure, reason in (
            (s.front_face_or_edge_hit, "front_face_or_edge_collision"),
            (s.rear_support_lost, "rear_support_lost"),
            (s.body_collision_or_fall, "body_collision_or_fall"),
            (s.deliberate_reverse, "deliberate_reverse"),
            (self.elapsed_s >= self.config.timeout_s, "timeout"),
        ):
            if failure:
                self.stage, self.reason = Stage.FAILED, reason
                return self.stage
        self.front_clearance = [a or b for a, b in zip(self.front_clearance, s.front_clearance_verified)]
        heading_error = math.atan2(math.sin(s.heading_rad), math.cos(s.heading_rad))
        aligned = (abs(s.lateral_m) <= self.config.line_tolerance_m
                   and abs(heading_error) <= self.config.heading_tolerance_rad)
        ready = (s.four_wheels_on_top and all(self.front_clearance) and aligned
                 and max(abs(x) for x in s.hip_error_rad) <= self.config.hip_tolerance_rad)
        if not ready:
            # Returning to the line/pose does not retain an invalid drive segment.
            self.drive_start_m = None
            self.stopped_s = 0.
            self.stage = Stage.CROSSING
            return self.stage
        if self.drive_start_m is None:
            self.drive_start_m = s.forward_m
        if s.forward_m - self.drive_start_m < self.config.post_climb_distance_m:
            self.stage = Stage.PLATFORM_DRIVE
            self.stopped_s = 0.
            return self.stage
        self.stage = Stage.STOPPING
        stopped = (s.speed_m_s <= self.config.stop_speed_m_s
                   and abs(s.yaw_rate_rad_s) <= self.config.stop_yaw_rate_rad_s)
        self.stopped_s = self.stopped_s + s.dt_s if stopped else 0.
        if self.stopped_s + 1e-12 >= self.config.stop_hold_s:
            self.stage = Stage.COMPLETE
        return self.stage
