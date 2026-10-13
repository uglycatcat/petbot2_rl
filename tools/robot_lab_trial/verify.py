#!/usr/bin/env python3
"""robot_lab 容器自检：不训练，只确认环境与任务注册可用。

用法（容器内）：/isaac-sim/python.sh /workspace/robot_lab_env/verify.py
"""
import importlib
import os
import sys
import traceback

print("=" * 70)
print("robot_lab 容器自检")
print("=" * 70)
print(f"python      : {sys.version.split()[0]}  ({sys.executable})")
print(f"cwd         : {os.getcwd()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置)')}")

print("\n[1] 关键包版本")
for mod in ("torch", "isaaclab", "isaaclab_rl", "rsl_rl", "robot_lab"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:14s} {getattr(m, '__version__', '(无 __version__)')}  -> {getattr(m, '__file__', '')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {mod:14s} 导入失败: {type(exc).__name__}: {exc}")

print("\n[2] GPU")
try:
    import torch

    print(f"  torch.cuda.is_available()={torch.cuda.is_available()}  "
          f"count={torch.cuda.device_count()}  name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
except Exception as exc:  # noqa: BLE001
    print(f"  失败: {exc}")

print("\n[3] 任务注册 + Go2W 环境创建（headless）")
try:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401  触发任务注册

    ids = sorted(i for i in gym.registry.keys() if i.startswith("RobotLab-"))
    print(f"  已注册 RobotLab-* 任务 {len(ids)} 个，其中含 Go2W 的：")
    for i in ids:
        if "Go2W" in i:
            print(f"    {i}")
    assert any("Go2W" in i for i in ids), "未找到 Go2W 任务"

    from isaaclab_tasks.utils import parse_env_cfg

    task = "RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0"
    if task not in gym.registry:
        task = next(i for i in ids if "Go2W" in i)
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=4)
    env = gym.make(task, cfg=cfg)
    obs, _ = env.reset()
    print(f"  环境创建成功: {task}  num_envs={env.unwrapped.num_envs}")
    print(f"  obs 形状 policy={tuple(obs['policy'].shape)}  action_dim={env.unwrapped.action_manager.total_action_dim}")
    print(f"  action terms: {env.unwrapped.action_manager.active_terms}")
    # 先打印结论再关 app：simulation_app.close() 会直接结束进程，之后的 print 不会输出
    print("\n结果：PASS", flush=True)
    env.close()
    simulation_app.close()
    os._exit(0)
except Exception:  # noqa: BLE001
    traceback.print_exc()
    print("\n结果：FAIL（见上面的 traceback）")
    sys.exit(1)
