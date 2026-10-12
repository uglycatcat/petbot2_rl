#!/usr/bin/env python3
"""离屏渲染 + 录 mp4 的可行性测试（与训练无关，只验证这台服务器能不能出画面/视频）。

做法：起一个最小交互场景（地面 + 两个下落的小方块 + 一台俯视相机），跑 N 帧物理，
把相机 rgb 帧写成 mp4。跑在观察容器里：

    docker exec recovery-vis /isaac-sim/python.sh /workspace/recovery_env/render_test.py \
        --headless --enable_cameras

输出：/workspace/recovery/logs/render_test/render_test.mp4（宿主机对应
reference_project/recovery/logs/render_test/），同时报首末帧差异，用来证明画面真的在变
（全黑或帧间完全相同就说明渲染没生效）。
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Sim 离屏渲染 + mp4 录制测试")
parser.add_argument("--out", default="/workspace/recovery/logs/render_test", help="输出目录")
parser.add_argument("--frames", type=int, default=90, help="帧数")
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--width", type=int, default=854)
parser.add_argument("--fps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import imageio  # noqa: E402
import numpy as np  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


@configclass
class RenderTestSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

    # 俯视相机（identity 四元数 = 沿 -Z 看下去）
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam",
        update_period=0.0,
        height=args.height,
        width=args.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.6), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )

    cube_red = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_red",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, 0.0, 1.1)),
    )
    cube_blue = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_blue",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.08, 1.4)),
    )


def main() -> int:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
    scene = InteractiveScene(RenderTestSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    cam = scene["camera"]
    dt = sim.get_physics_dt()

    # 相机摆位：从斜上方看向原点（identity 四元数在这套约定下是水平朝向，拍不到地上的方块）
    import torch

    cam.set_world_poses_from_view(
        torch.tensor([[1.05, 1.05, 0.75]], device=sim.device),
        torch.tensor([[0.0, 0.0, 0.12]], device=sim.device),
    )

    frames = []
    for _ in range(args.frames):
        sim.step()
        scene.update(dt)
        rgb = cam.data.output["rgb"]
        img = rgb[0] if rgb.ndim == 4 else rgb
        frames.append(img.cpu().numpy().astype(np.uint8))

    a, b = frames[0].astype(np.int16), frames[-1].astype(np.int16)
    diff = float(np.abs(a - b).mean())
    os.makedirs(args.out, exist_ok=True)
    mp4 = os.path.join(args.out, "render_test.mp4")
    imageio.mimsave(mp4, frames, fps=args.fps)
    imageio.imwrite(os.path.join(args.out, "frame_000.png"), frames[0])
    imageio.imwrite(os.path.join(args.out, "frame_last.png"), frames[-1])

    print("== 渲染结果 ==", flush=True)
    print(f"分辨率      : {frames[0].shape}", flush=True)
    print(f"帧数        : {len(frames)}", flush=True)
    print(f"首帧亮度均值: {frames[0].mean():.1f}  (0 = 全黑)", flush=True)
    print(f"首末帧平均差: {diff:.2f}  (0 = 画面没变)", flush=True)
    print(f"mp4         : {mp4}  {os.path.getsize(mp4) / 1024:.0f} KB", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    # 实测：在 --enable_cameras 下 app.close()（Kit 渲染器退出）会长时间挂住，
    # 而这个脚本只是个渲染能力测试，产物已落盘，直接退出更省事。
    # 训练脚本结束时若也卡在关闭阶段，同样可以 kill —— 检查点在此之前已经写完。
    sys.stdout.flush()
    os._exit(code)
