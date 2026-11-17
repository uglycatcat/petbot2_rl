"""Headless fallen-pose loading for the unchanged PetBot MJCF scene.

This module starts no viewer or simulation loop and does not configure GLFW.
See README.md for the four recorded pose IDs and the reset contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
SCENE = HERE.parents[2] / "robot_description/mjcf/scene.xml"
SNAPSHOTS = HERE / "snapshots"
FORMAT = "petbot-pose-snapshot-v2"
LEGACY_FORMAT = "petbot-pose-snapshot-v1"
FLOOR_CLEARANCE_M = 0.001


def recorded_pose_path(pose_id: str) -> Path:
    """Resolve pose_01..pose_04 and verify the original JSON's SHA-256."""
    manifest = json.loads((SNAPSHOTS / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["poses"]:
        if entry["id"] == pose_id:
            path = SNAPSHOTS / entry["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"重要原始姿态校验失败：{pose_id} ({path.name})")
            return path
    available = ", ".join(entry["id"] for entry in manifest["poses"])
    raise ValueError(f"未知姿态 {pose_id!r}；可选：{available}")


class PoseLoader:
    """Load into a new or caller-owned model/data pair without running physics.

    Supplied model/data must use the unchanged SCENE and data must belong to that
    model. Loading resets all dynamic state; the caller owns the control loop.
    """

    def __init__(self, model: mujoco.MjModel | None = None,
                 data: mujoco.MjData | None = None):
        if data is not None and model is None:
            raise ValueError("传入 data 时必须同时传入创建它的 model")
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = data if data is not None else mujoco.MjData(self.model)
        self.root_qadr = int(self.model.joint("root").qposadr[0])
        self.floor_id = self.model.geom("floor").id
        self.robot_geoms = np.flatnonzero(self.model.geom_bodyid != 0)
        digest = hashlib.sha256()
        for path in (SCENE, SCENE.with_name("robot_description.xml")):
            digest.update(path.read_bytes())
        self.signature = {
            "mjcf_sha256": digest.hexdigest(),
            "nq": self.model.nq,
            "nv": self.model.nv,
            "joints": [self.model.joint(i).name for i in range(self.model.njnt)],
        }

    def floor_clearance(self, data=None) -> float:
        """Signed distance from the robot's collision geometry to the floor."""
        if data is None:
            data = self.data
        distmax = max(1.0, abs(float(data.qpos[self.root_qadr + 2])) + 1.0)
        return min(float(mujoco.mj_geomDistance(
            self.model, data, self.floor_id, int(geom), distmax, None))
            for geom in self.robot_geoms)

    def prepare_fallen_pose(self, qpos):
        """Center horizontally, preserve rotation/joints, and only raise height.

        Use separate data so saving or a failed load cannot move the live robot.
        The unchanged source scene has a horizontal floor at world z=0.
        """
        root = self.root_qadr
        candidate = mujoco.MjData(self.model)
        candidate.qpos[:] = qpos
        candidate.qpos[root:root + 2] = self.model.qpos0[root:root + 2]
        source_height = float(candidate.qpos[root + 2])
        mujoco.mj_forward(self.model, candidate)
        distance = self.floor_clearance(candidate)
        if not np.isfinite(distance):
            raise ValueError("无法计算该姿态与地面的距离")
        if distance < FLOOR_CLEARANCE_M - 1e-9:
            candidate.qpos[root + 2] += FLOOR_CLEARANCE_M - distance
            mujoco.mj_forward(self.model, candidate)
        distance = self.floor_clearance(candidate)
        if not np.isfinite(distance) or distance < FLOOR_CLEARANCE_M - 1e-9:
            raise ValueError("该姿态无法安全放置到地面上方；当前姿态未改变")
        height = float(candidate.qpos[root + 2])
        return candidate.qpos.copy(), {
            "source_height_m": source_height,
            "height_m": height,
            "height_adjustment_m": height - source_height,
            "floor_clearance_m": distance,
        }

    def load(self, path: str | Path) -> dict:
        """Reset data to a centered, floor-safe pose; return placement in meters.

        No integration step, controller, threads or GUI is started. Original
        files are read only. Validation/placement failures leave data unchanged.
        """
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("format") not in (FORMAT, LEGACY_FORMAT):
            raise ValueError("不是本 demo 的姿态快照")
        if record.get("model") != self.signature:
            raise ValueError("快照的 MJCF 或关节布局与当前模型不一致")
        qpos = np.asarray(record.get("qpos"), dtype=float)
        if qpos.shape != (self.model.nq,) or not np.all(np.isfinite(qpos)):
            raise ValueError("快照 qpos 的长度或数值无效")
        quat = qpos[self.root_qadr + 3:self.root_qadr + 7]
        if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6, rtol=0):
            raise ValueError("快照中的机身四元数必须是单位四元数")
        qpos, placement = self.prepare_fallen_pose(qpos)
        # Validate and check clearance before touching the caller's live data.
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return placement
