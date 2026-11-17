"""Read the four protected recordings; no integration or source-file writes."""
import hashlib
import json
import sys

import numpy as np

from .model import PROJECT_ROOT, workspace_root


def load_poses():
    root = workspace_root()
    sys.path.insert(0, str(root))
    from robot_controller_demo.test.pose_snapshot_demo.pose_io import PoseLoader, recorded_pose_path
    cfg = json.loads((PROJECT_ROOT / 'configs/initial_poses.json').read_text())
    loader = PoseLoader()
    rows, reports = [], []
    for entry in cfg['poses']:
        path = recorded_pose_path(entry['id'])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError(f"Recorded pose changed: {entry['id']}")
        report = loader.load(path)
        rows.append(loader.data.qpos.copy())
        reports.append(dict(id=entry['id'], sha256=entry['sha256'], **report))
    qadr = {loader.model.joint(i).name: int(loader.model.jnt_qposadr[i])
            for i in range(loader.model.njnt)}
    return np.asarray(rows), qadr, reports
