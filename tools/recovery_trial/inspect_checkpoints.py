#!/usr/bin/env python3
"""对比作者公开权重与我们训练检查点的张量结构（判断能否互相加载）。"""
import torch

PATHS = {
    "author_model_7999": "/workspace/recovery_env/models/model_7999.pt",
    "author_model_4800": "/workspace/recovery_env/models/model_4800.pt",
    "ours_model_9999": "/workspace/recovery/logs/rsl_rl/thunder_recovery/2026-10-09_10-42-27/model_9999.pt",
}


def show(name: str, path: str) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n=== {name} ===")
    print("类型:", type(ck).__name__, end="")
    if isinstance(ck, dict):
        print("，顶层键:", list(ck.keys()))
    else:
        print()
        return
    sd = ck.get("model_state_dict", ck)
    keys = list(sd.keys())
    print(f"参数张量数: {len(keys)}")
    for k in keys:
        v = sd[k]
        if hasattr(v, "shape") and v.ndim >= 1 and any(s > 1 for s in v.shape):
            print(f"   {k:<42} {tuple(v.shape)}")


for n, p in PATHS.items():
    try:
        show(n, p)
    except Exception as e:  # noqa: BLE001
        print(f"\n=== {n} === 载入失败: {e}")
