"""Load one recorded pose without opening a window or advancing simulation."""

import argparse
import json
from pathlib import Path

if __package__:
    from .pose_io import PoseLoader, recorded_pose_path
else:
    from pose_io import PoseLoader, recorded_pose_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pose", default="pose_01", choices=[f"pose_{i:02d}" for i in range(1, 5)])
    source.add_argument("--json", type=Path, help="加载指定 v1/v2 快照，而不是内置姿态ID")
    args = parser.parse_args()
    path = args.json if args.json is not None else recorded_pose_path(args.pose)

    loader = PoseLoader()
    placement = loader.load(path)
    model, data = loader.model, loader.data
    # Hand model/data to your own controller or simulator loop here.
    # No mj_step is called: the loaded initial pose is still intact.
    root = int(model.joint("root").qposadr[0])
    print(json.dumps({
        "file": path.name,
        "root_xyz_m": data.qpos[root:root + 3].tolist(),
        "root_quat_wxyz": data.qpos[root + 3:root + 7].tolist(),
        "placement": placement,
        "sim_time_s": data.time,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
