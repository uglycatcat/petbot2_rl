"""Read-only source model and explicit control contract (no simulator dependency)."""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSITION_JOINTS = ("head", "left_ear", "right_ear", "left_frame", "right_frame")
REAR_WHEELS = ("left_rear_wheel", "right_rear_wheel")
FRONT_WHEELS = ("left_front_wheel", "right_front_wheel")
WHEELS = FRONT_WHEELS + REAR_WHEELS
JOINT_ORDER = POSITION_JOINTS + REAR_WHEELS + FRONT_WHEELS
# Initial engineering values, not measured motor gains. Torque caps come from MJCF.
POSITION_KP = dict(zip(POSITION_JOINTS, (3.0, 1.5, 1.5, 4.0, 4.0), strict=True))
POSITION_KD = dict(zip(POSITION_JOINTS, (0.10, 0.035, 0.035, 0.12, 0.12), strict=True))
ADAPTER_VERSION = 1


def source_dir() -> Path:
    override = os.environ.get("PETBOT_MODEL_DIR")
    candidates = [Path(override)] if override else [p / "robot_description" for p in PROJECT_ROOT.parents]
    for path in candidates:
        if (path / "mjcf/robot_description.xml").is_file():
            return path.resolve()
    raise FileNotFoundError("Mount the workspace including robot_description, or set PETBOT_MODEL_DIR.")


def source_fingerprint(directory: Path | None = None) -> str:
    directory = directory or source_dir()
    digest = hashlib.sha256()
    paths = sorted([*directory.glob("mjcf/*.xml"), *directory.glob("meshes/*")])
    for path in paths:
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_model(directory: Path | None = None) -> dict:
    directory = directory or source_dir()
    root = ET.parse(directory / "mjcf/robot_description.xml").getroot()
    base_defaults = root.find("default/joint").attrib
    wheel_defaults = root.find("default/default[@class='wheel']/joint").attrib
    joints = {}
    for joint in root.findall("worldbody//joint"):
        attrs = dict(base_defaults)
        if joint.get("class") == "wheel":
            attrs.update(wheel_defaults)
        attrs.update(joint.attrib)
        name = attrs["name"]
        joints[name] = {
            "axis": [float(x) for x in attrs["axis"].split()],
            "range": [float(x) for x in attrs["range"].split()] if attrs.get("limited") != "false" else None,
            "default": float(attrs.get("ref", 0)),
            "damping": float(attrs.get("damping", 0)),
            "armature": float(attrs.get("armature", 0)),
            "friction": float(attrs.get("frictionloss", 0)),
        }
    if set(joints) != set(JOINT_ORDER):
        raise ValueError(f"Unexpected source joints: {list(joints)}")
    actuators = {}
    for actuator in root.find("actuator"):
        attrs = actuator.attrib
        actuators[attrs["joint"]] = {
            "type": actuator.tag,
            "effort": max(abs(float(x)) for x in attrs["forcerange"].split()),
            "control_range": [float(x) for x in attrs["ctrlrange"].split()],
            "kv": float(attrs.get("kv", 0)),
        }
    if set(actuators) != set(POSITION_JOINTS + REAR_WHEELS):
        raise ValueError("Source must contain exactly five position candidates and two rear wheel drives.")
    return {
        "source_sha256": source_fingerprint(directory),
        "joints": joints,
        "actuators": actuators,
        "mass": sum(float(x.attrib["mass"]) for x in root.findall("worldbody//inertial")),
        "root_position": [float(x) for x in root.find("worldbody/body").get("pos").split()],
        "physics_dt": float(root.find("option").get("timestep")),
    }


def asset_identity() -> str:
    payload = {
        "source": source_fingerprint(),
        "adapter": ADAPTER_VERSION,
        "kp": POSITION_KP,
        "kd": POSITION_KD,
        "converter": hashlib.sha256(Path(__file__).with_name("prepare.py").read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def asset_dir() -> Path:
    return PROJECT_ROOT / "artifacts/petbot_asset" / asset_identity()


def prepared_asset() -> tuple[Path, dict]:
    directory = asset_dir()
    metadata = directory / "asset.json"
    if not metadata.is_file():
        raise FileNotFoundError(
            "Prepare the asset in the training container: python -m agile.petbot.prepare --headless"
        )
    manifest = json.loads(metadata.read_text())
    usd = directory / manifest["usd"]
    if manifest["source_sha256"] != source_fingerprint() or not usd.is_file():
        raise ValueError("Prepared PetBot asset is missing or stale. Run agile.petbot.prepare again.")
    return usd, manifest


def task_implementation_fingerprint() -> str:
    """Behavioral code identity independent of documentation-only commits."""
    names = ("core.py", "mdp.py", "env_cfg.py", "asset_cfg.py", "collection.py", "distribution.py", "curriculum.py", "agents.py", "pre_learn.py")
    payload = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
