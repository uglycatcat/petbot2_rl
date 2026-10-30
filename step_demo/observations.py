"""One initial terrain packet plus own sensor estimates form the actor input.

No contact labels, true pose, current terrain distance or evaluator state are
accepted. The estimator itself is not yet implemented or physically validated.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TerrainPacket:
    height_m: float
    distance_m: float
    edge_angle_rad: float  # tangent angle from initial forward, modulo pi

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in (self.height_m, self.distance_m, self.edge_angle_rad))
                or self.height_m <= 0 or self.distance_m <= 0
                or not 0 < self.edge_angle_rad < math.pi):
            raise ValueError("Invalid initial terrain packet")


@dataclass(frozen=True)
class MotionEstimate:
    """Own-motion estimate in the initial COM/heading frame, never ground truth."""
    forward_m: float
    lateral_m: float
    heading_rad: float
    forward_speed_m_s: float
    lateral_speed_m_s: float


@dataclass(frozen=True)
class Proprioception:
    gravity_body: tuple[float, float, float]
    gyro_rad_s: tuple[float, float, float]
    hip_position_rad: tuple[float, float]
    hip_velocity_rad_s: tuple[float, float]
    rear_velocity_rad_s: tuple[float, float]


OBSERVATION_FIELDS = (
    "terrain_height", "initial_edge_distance", "initial_edge_sin", "initial_edge_cos",
    "estimated_normal_distance", "estimated_edge_sin", "estimated_edge_cos",
    "estimated_lateral_position", "estimated_heading_sin", "estimated_heading_cos",
    "estimated_forward_speed", "estimated_lateral_speed",
    "gravity_x", "gravity_y", "gravity_z", "gyro_x", "gyro_y", "gyro_z",
    "left_hip_position", "right_hip_position", "left_hip_velocity", "right_hip_velocity",
    "left_rear_velocity", "right_rear_velocity", "previous_hip_action", "previous_wheel_action",
)


class ObservationBuilder:
    def __init__(self):
        self._terrain: TerrainPacket | None = None

    def reset(self):
        """Only call at a new physical episode, never to refresh terrain mid-run."""
        self._terrain = None

    def receive_initial_terrain(self, packet: TerrainPacket):
        if self._terrain is not None:
            raise RuntimeError("Terrain perception may be supplied only once per episode")
        self._terrain = packet

    def build(self, sensors: Proprioception, estimate: MotionEstimate,
              previous_action: tuple[float, float]) -> tuple[float, ...]:
        if self._terrain is None:
            raise RuntimeError("Initial terrain packet required")
        groups = ((sensors.gravity_body, 3), (sensors.gyro_rad_s, 3),
                  (sensors.hip_position_rad, 2), (sensors.hip_velocity_rad_s, 2),
                  (sensors.rear_velocity_rad_s, 2), (previous_action, 2))
        if any(len(values) != size for values, size in groups):
            raise ValueError("Each sensor group must match its declared dimension")
        t = self._terrain
        # Edge through (initial distance, 0), with normal pointing into the platform.
        distance = ((t.distance_m - estimate.forward_m) * math.sin(t.edge_angle_rad)
                    + estimate.lateral_m * math.cos(t.edge_angle_rad))
        angle = t.edge_angle_rad - estimate.heading_rad
        values = (t.height_m, t.distance_m, math.sin(t.edge_angle_rad), math.cos(t.edge_angle_rad),
                  distance, math.sin(angle), math.cos(angle), estimate.lateral_m,
                  math.sin(estimate.heading_rad), math.cos(estimate.heading_rad),
                  estimate.forward_speed_m_s, estimate.lateral_speed_m_s,
                  *sensors.gravity_body, *sensors.gyro_rad_s, *sensors.hip_position_rad,
                  *sensors.hip_velocity_rad_s, *sensors.rear_velocity_rad_s, *previous_action)
        if len(values) != len(OBSERVATION_FIELDS) or not all(math.isfinite(x) for x in values):
            raise ValueError("Observation shape or finite-value contract violated")
        if any(abs(x) > 1 for x in previous_action):
            raise ValueError("Previous action must be the two executed normalized targets")
        return values
