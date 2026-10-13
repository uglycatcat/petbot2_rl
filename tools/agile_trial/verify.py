#!/usr/bin/env python3
"""WBC-AGILE 容器自检：不训练，只确认环境与 StandUp-T1-v0 任务可用。

用法（容器内）：/isaac-sim/python.sh /workspace/agile_env/verify.py
"""
import importlib
import os
import sys
import traceback

print("=" * 70)
print("WBC-AGILE 容器自检")
print("=" * 70)
print(f"python      : {sys.version.split()[0]}  ({sys.executable})")
print(f"cwd         : {os.getcwd()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置)')}")

print("\n[1] 关键包版本（上游要求 isaaclab 3.0.0b2 / python 3.12 / rsl-rl-lib 5.4.1）")
for mod in ("torch", "isaaclab", "isaaclab_rl", "rsl_rl", "agile", "mujoco", "leapp"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:12s} {getattr(m, '__version__', '(无 __version__)')}  -> {getattr(m, '__file__', '')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {mod:12s} 导入失败: {type(exc).__name__}: {exc}")

print("\n[2] GPU")
try:
    import torch

    ok = torch.cuda.is_available()
    print(f"  torch.cuda.is_available()={ok}  count={torch.cuda.device_count()}  "
          f"name={torch.cuda.get_device_name(0) if ok else '-'}")
except Exception as exc:  # noqa: BLE001
    print(f"  失败: {exc}")

print("\n[3] 任务注册 + StandUp-T1-v0 环境创建（headless）")
try:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import agile.rl_env.tasks  # noqa: F401  触发任务注册

    ids = sorted(i for i in gym.registry.keys())
    print(f"  已注册任务 {len(ids)} 个：{ids}")

    task = "StandUp-T1-v0"
    assert task in ids, f"{task} 未注册"

    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(task, device="cuda:0", num_envs=4)
    env = gym.make(task, cfg=cfg)
    obs, _ = env.reset()
    print(f"  环境创建成功: {task}  num_envs={env.unwrapped.num_envs}")

    def describe(o, indent="    "):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, dict):
                    print(f"{indent}{k}:")
                    describe(v, indent + "  ")
                else:
                    print(f"{indent}{k}: {tuple(v.shape) if hasattr(v, 'shape') else type(v)}")
        else:
            print(f"{indent}{tuple(o.shape) if hasattr(o, 'shape') else type(o)}")

    print("  obs 结构（AGILE 的 ObsGroup 用 concatenate_terms=False，所以是嵌套 dict）:")
    describe(obs)
    am = env.unwrapped.action_manager
    print(f"  action_dim={am.total_action_dim}  action terms={am.active_terms}")
    print(f"  终止项: {env.unwrapped.termination_manager.active_terms}")
    print(f"  课程项: {env.unwrapped.curriculum_manager.active_terms}")
    # 先打印结论再关 app：simulation_app.close() 会直接结束进程，之后的 print 不会输出
    print("\n结果：PASS", flush=True)
    env.close()
    simulation_app.close()
    os._exit(0)
except Exception:  # noqa: BLE001
    traceback.print_exc()
    print("\n结果：FAIL（见上面的 traceback）")
    sys.exit(1)
