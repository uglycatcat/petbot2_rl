"""Read source limits without changing MJCF, meshes or physical parameters."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class ModelLimits:
    hip_min_rad: float
    hip_max_rad: float
    rear_speed_max_rad_s: float
    hip_effort_max_nm: float
    rear_effort_max_nm: float
    physics_dt_s: float


def _range(node: ET.Element | None, name: str) -> tuple[float, float]:
    if node is None:
        raise ValueError(f"Missing source element for {name}")
    values = tuple(float(x) for x in node.attrib[name].split())
    if len(values) != 2 or not all(math.isfinite(x) for x in values) or values[0] >= values[1]:
        raise ValueError(f"Invalid source {name}: {values}")
    return values


def read_limits(model_dir: Path) -> ModelLimits:
    root = ET.parse(model_dir / "mjcf/robot_description.xml").getroot()
    hips, wheels, hip_efforts, wheel_efforts = [], [], [], []
    for side in ("left", "right"):
        hips.append(_range(root.find(f".//joint[@name='{side}_frame']"), "range"))
        motor = root.find(f"./actuator/motor[@joint='{side}_frame']")
        hip_efforts.append(_range(motor, "forcerange"))
        velocity = root.find(f"./actuator/velocity[@joint='{side}_rear_wheel']")
        wheels.append(_range(velocity, "ctrlrange"))
        wheel_efforts.append(_range(velocity, "forcerange"))
    for pairs in (hips, wheels, hip_efforts, wheel_efforts):
        if pairs[0] != pairs[1]:
            raise ValueError("Symmetric demo requires matching left/right source limits")
    if not hips[0][0] < 0 < hips[0][1] or wheels[0][1] <= 0:
        raise ValueError("Source must support default hip pose and forward wheel targets")
    # Position targets need an actuator adapter; source hip actuators are torque motors.
    option = root.find("option")
    if option is None:
        raise ValueError("Missing source physics options")
    dt = float(option.attrib["timestep"])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid source physics timestep")
    return ModelLimits(hips[0][0], hips[0][1], min(25., wheels[0][1]),
                       min(-hip_efforts[0][0], hip_efforts[0][1]),
                       min(-wheel_efforts[0][0], wheel_efforts[0][1]), dt)


def source_fingerprint(model_dir: Path) -> str:
    """Hash source XML/STL bytes and relative names; never hash derived assets."""
    files = sorted(p for p in model_dir.rglob("*") if p.is_file()
                   and p.suffix.lower() in (".xml", ".stl"))
    if not files:
        raise ValueError("No source XML/STL files found")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(model_dir).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
