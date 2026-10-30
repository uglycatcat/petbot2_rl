"""Load the fixed first-demo contract; acceptance values are provisional."""

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/step_v1.json"


@dataclass(frozen=True)
class TerrainConfig:
    height_m: float
    initial_distance_m: float
    edge_angle_rad: float
    platform_length_m: float
    platform_width_m: float


@dataclass(frozen=True)
class ControlConfig:
    default_hip_rad: float
    head_rad: float
    left_ear_rad: float
    right_ear_rad: float


@dataclass(frozen=True)
class AcceptanceConfig:
    post_climb_distance_m: float
    line_tolerance_m: float
    heading_tolerance_rad: float
    hip_tolerance_rad: float
    stop_speed_m_s: float
    stop_yaw_rate_rad_s: float
    stop_hold_s: float
    timeout_s: float


@dataclass(frozen=True)
class StepConfig:
    model_dir: Path
    terrain: TerrainConfig
    control: ControlConfig
    acceptance: AcceptanceConfig


def load_config(path: Path = DEFAULT_CONFIG, model_dir: Path | None = None) -> StepConfig:
    data = json.loads(Path(path).read_text())
    expected = {"schema", "model_dir", "terrain", "control", "acceptance"}
    if set(data) != expected or data["schema"] != "petbot_step_v1":
        raise ValueError("Expected petbot_step_v1 with exactly the documented fields")
    terrain = TerrainConfig(**data["terrain"])
    control = ControlConfig(**data["control"])
    acceptance = AcceptanceConfig(**data["acceptance"])
    for group in (terrain, control, acceptance):
        if not all(isinstance(getattr(group, f.name), (int, float))
                   and not isinstance(getattr(group, f.name), bool)
                   and math.isfinite(getattr(group, f.name)) for f in fields(group)):
            raise ValueError("All contract values must be finite numbers")
    if (terrain.height_m != 0.02 or terrain.initial_distance_m != 0.30
            or not math.isclose(terrain.edge_angle_rad, math.pi / 2, abs_tol=1e-12)):
        raise ValueError("First demo is fixed at 2 cm, 30 cm, and 90 degrees")
    if terrain.platform_length_m <= 0 or terrain.platform_width_m <= 0:
        raise ValueError("Platform extents must be positive")
    if any(getattr(acceptance, f.name) <= 0 for f in fields(acceptance)):
        raise ValueError("Acceptance distances, tolerances and durations must be positive")
    if acceptance.stop_hold_s >= acceptance.timeout_s:
        raise ValueError("Stop hold must fit inside the episode timeout")
    if any(getattr(control, f.name) != 0 for f in fields(control)):
        raise ValueError("First demo uses the source default joint pose (zero)")
    source = Path(model_dir) if model_dir is not None else PROJECT_ROOT / data["model_dir"]
    return StepConfig(source.resolve(), terrain, control, acceptance)
