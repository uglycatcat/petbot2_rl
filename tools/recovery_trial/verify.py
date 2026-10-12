#!/usr/bin/env python3
"""Kitjesen/recovery 容器内环境自检（不训练、不建环境实例）。

容器里跑：
    /isaac-sim/python.sh /workspace/recovery_env/verify.py            # 含 Isaac Sim 启动检查
    /isaac-sim/python.sh /workspace/recovery_env/verify.py --skip-app # 只查版本与源码侧 API

检查项：
  1) 关键包版本（python / torch / isaaclab / isaaclab_rl / rsl-rl-lib）
  2) 上游 scripts/train.py 依赖的 API 是否存在于已安装的 isaaclab 源码里
     （用文件扫描而不是 import —— 未启动 App 前 `import isaaclab` 会因缺 pxr 失败）
  3) Isaac Sim 能否 headless 启动、两个 gym task 是否注册成功
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
import urllib.parse


def version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except Exception:  # noqa: BLE001
        return "(未安装)"


def isaaclab_source_root() -> str | None:
    """解析 isaaclab 的安装位置（镜像内是 editable 装到 /workspace/isaaclab/source/isaaclab）。"""
    try:
        dist = md.distribution("isaaclab")
        raw = dist.read_text("direct_url.json")
        if raw:
            url = json.loads(raw).get("url", "")
            if url.startswith("file://"):
                return urllib.parse.unquote(url[len("file://"):])
        return str(dist.locate_file(""))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-app", action="store_true", help="不启动 Isaac Sim")
    args = ap.parse_args()

    print("== 1. 版本 ==")
    print("python            :", sys.version.split()[0])
    for pkg in ("torch", "numpy", "gymnasium", "isaaclab", "isaaclab_rl", "isaaclab_tasks", "rsl-rl-lib"):
        print(f"{pkg:<18}: {version(pkg)}")

    print("\n== 2. 上游 train.py 依赖的 API（扫描已安装源码）==")
    root = isaaclab_source_root()
    print("isaaclab 安装位置 :", root)
    if root:
        io_init = f"{root}/isaaclab/utils/io/__init__.py"
        try:
            txt = open(io_init).read()
            for name in ("dump_yaml", "dump_pickle"):
                print(f"  {io_init} 含 {name}: {name in txt}")
            if "dump_pickle" not in txt:
                print("  → 上游 scripts/train.py 的 `from isaaclab.utils.io import dump_pickle` 在该版本缺失")
        except OSError as e:
            print("  读取失败:", e)

    if args.skip_app:
        return 0

    print("\n== 3. Isaac Sim 启动 + 任务注册 ==")
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        import gymnasium as gym
        import thunder_recovery  # noqa: F401  触发 gym.register

        try:
            sim_ver = open("/isaac-sim/VERSION").read().strip()
        except OSError:
            sim_ver = "(未知)"
        print("isaac sim 版本   :", sim_ver)
        print("已注册 Recovery 任务:", [k for k in gym.registry if "Recovery-Thunder" in k])
        print("Isaac Sim 启动成功")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
