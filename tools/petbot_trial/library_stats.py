"""统计初态库的构成：直立/高度/构型/低速 四项（不含接触支撑）在 t=0 时的通过率。

用来判断"评估成功率"里有多少来自"落地时已经接近可站住"的样本。
用法（容器内）：/isaac-sim/python.sh /workspace/demo_env/library_stats.py <副本目录>/artifacts/fallen_states/*.pt
"""

import glob
import math
import os
import sys

import torch

TARGET_HEIGHT = 0.084271


def analyze(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    d = payload["states"]
    root, q, v = d["root"], d["joint_pos"], d["joint_vel"]
    w, x, y, z = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    vertical = 1 - 2 * (x * x + y * y)
    upright = vertical >= math.cos(math.radians(15))
    height = (root[:, 2] - TARGET_HEIGHT).abs() <= 0.02
    configuration = q[:, :5].abs().amax(1) <= 0.15
    still = (
        (v[:, :5].abs().amax(1) <= 0.15)
        & (v[:, 5:].abs().amax(1) <= 0.5)
        & (root[:, 7:10].norm(dim=1) <= 0.05)
        & (root[:, 10:13].norm(dim=1) <= 0.15)
    )
    all4 = upright & height & configuration & still
    print(f"{os.path.basename(path):<44} n={len(root):<6}")
    print(f"  直立 {upright.float().mean():.3f} | 高度 {height.float().mean():.3f} | 构型 {configuration.float().mean():.3f} | 低速 {still.float().mean():.3f} | 四项同时 {all4.float().mean():.3f}")
    print(f"  姿态: 倒置 {(vertical < -0.7).float().mean():.3f} 侧躺 {((vertical >= -0.7) & (vertical <= 0.7)).float().mean():.3f} 近直立 {(vertical > 0.7).float().mean():.3f}")
    print(f"  高度: min {root[:, 2].min():.3f} p50 {root[:, 2].median():.3f} max {root[:, 2].max():.3f}")


for path in sorted(sum((glob.glob(p) for p in sys.argv[1:]), [])):
    analyze(path)
