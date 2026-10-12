#!/usr/bin/env python3
"""用训练检查点评估 recovery 策略的真实成功率（无视频，纯指标）。

背景：训练日志里的 `Episode_Reward/recovery_success_rate` 是 `1e-6` 权重的纯日志项，
加权后打印 4 位小数永远显示 0.0000，**无法反映真实成功率**，所以必须离线评估。

本脚本同时统计两套判据：
  A. 上游 `check_recovery_success` 原样：高度 >0.30 m、**全部 16 个关节**的偏差 <0.5 rad、
     全部关节 max|q̇| <0.1 rad/s、姿态 |g_b−[0,0,-1]| <0.1。
  B. 只看 12 个腿关节的同两条（轮子是 continuous 关节、绝对角无限累积，
     把轮角算进"偏差 <0.5 rad"等于永不达标 —— 用于区分"判据缺陷"与"策略没学会"）。
另外报告"高度+姿态达标"（能否站起来）与各阈值下的分位数分布。

运行：
    docker exec recovery-vis /isaac-sim/python.sh /workspace/recovery_env/eval_recovery.py \
        --checkpoint .../model_9999.pt --num_envs 2048 --episodes 3

episode 为 250 个控制步（50 Hz × 5 s）；第 250 步触发 time_out 自动重置，
故有效窗口取 0..248，"最后 1 秒"= 第 199..248 步。
"""
from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="离线评估 recovery 策略成功率")
parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Recovery-Thunder-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="/workspace/recovery/logs/eval_recovery.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from thunder_recovery.config.method1_deng.env_cfg import LEG_JOINT_NAMES  # noqa: E402

T_VALID = 249
LAST_SEC = 50
CONDS = ["h30", "j_all", "v_all", "o", "j_leg", "v_leg", "h42"]


def main() -> int:
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    env = gym.make(args.task, cfg=env_cfg)
    vec = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    unwrapped = vec.unwrapped
    asset = unwrapped.scene["robot"]
    cfg = SceneEntityCfg("robot")
    dev = unwrapped.device
    n = args.num_envs

    # episode 长度与控制周期由 env 决定（论文口径 4 s @ 50 Hz = 200 控制步；
    # 上游原配置是 5 s = 250 步）。最后一步会触发 time_out 自动重置，故排除。
    T_VALID = int(unwrapped.max_episode_length) - 1
    LAST_SEC = int(round(1.0 / float(unwrapped.step_dt)))
    print(f"评估口径: episode {int(unwrapped.max_episode_length)} 控制步, "
          f"{1.0/float(unwrapped.step_dt):.0f} Hz, 有效窗口 {T_VALID} 步, 最后 1 秒 = {LAST_SEC} 步",
          flush=True)

    names = list(asset.joint_names)
    leg_idx = [names.index(j) for j in LEG_JOINT_NAMES if j in names]
    print(f"关节总数 {len(names)}，腿关节索引 {len(leg_idx)} 个：{leg_idx}", flush=True)
    lidx = torch.tensor(leg_idx, device=dev, dtype=torch.long)
    ideal = torch.tensor([0.0, 0.0, -1.0], device=dev)

    ok_all, g0_all, cond_all, diag_all = [], [], [], []
    for ep in range(args.episodes):
        obs, _ = vec.reset()
        g0_all.append(asset.data.projected_gravity_b.clone().cpu())
        conds = torch.zeros(n, T_VALID, len(CONDS), dtype=torch.bool)
        for t in range(T_VALID):
            with torch.no_grad():
                actions = policy(obs)
                obs, _, _, _ = vec.step(actions)
            a = asset
            jdev_all = torch.norm(a.data.joint_pos - a.data.default_joint_pos, dim=1)
            vmax_all = torch.max(torch.abs(a.data.joint_vel), dim=1).values
            jdev_leg = torch.norm(a.data.joint_pos[:, lidx] - a.data.default_joint_pos[:, lidx], dim=1)
            vmax_leg = torch.max(torch.abs(a.data.joint_vel[:, lidx]), dim=1).values
            conds[:, t] = torch.stack([
                a.data.root_pos_w[:, 2] > 0.30,
                jdev_all < 0.5,
                vmax_all < 0.1,
                torch.norm(a.data.projected_gravity_b - ideal, dim=1) < 0.1,
                jdev_leg < 0.5,
                vmax_leg < 0.1,
                a.data.root_pos_w[:, 2] > 0.42,   # 论文平地判据的高度阈
            ], dim=1).cpu()
        cond_all.append(conds)
        diag_all.append({
            "height": a.data.root_pos_w[:, 2].cpu().numpy(),
            "ori_err": torch.norm(a.data.projected_gravity_b - ideal, dim=1).cpu().numpy(),
            "jdev_all": jdev_all.cpu().numpy(),
            "jdev_leg": jdev_leg.cpu().numpy(),
            "vmax_all": vmax_all.cpu().numpy(),
            "vmax_leg": vmax_leg.cpu().numpy(),
        })

    conds = torch.cat(cond_all)              # (S, T, 6)
    g0 = torch.cat(g0_all)
    S = conds.shape[0]
    tail = conds[:, -LAST_SEC:, :]

    def suite(mask_idx: list[int]) -> torch.Tensor:
        """最后 1 秒内，指定条件全程成立的 per-step 与 per-sample 结果。"""
        sub = tail[:, :, mask_idx].all(dim=2)          # (S, 50) 每步是否满足该套判据
        return sub, sub.all(dim=1)                     # per-step, per-sample(严格)

    step_all, strict_all = suite([0, 1, 2, 3])         # A: 上游原样
    step_leg, strict_leg = suite([0, 4, 5, 3])         # B: 腿关节版
    step_paper, strict_paper = suite([6, 4, 5])        # C: 论文平地判据（高度>0.42 + 腿关节）
    stand = (conds[:, -1, 0] & conds[:, -1, 3])        # 最后一步：高度 + 姿态达标

    # 恢复时刻（严格口径 A / B）
    def recovery_time(strict_ok_steps: torch.Tensor) -> torch.Tensor:
        ok = strict_ok_steps
        rev = torch.full((S,), -1.0)
        running = torch.ones(S, dtype=torch.bool)
        for t in range(ok.shape[1] - 1, -1, -1):
            running &= ok[:, t]
            rev[running] = t
        rev[rev < 0] = float("nan")
        return rev

    rev_a = recovery_time(step_all)
    rev_b = recovery_time(step_leg)
    rev_c = recovery_time(step_paper)

    gz, gx, gy = g0[:, 2].numpy(), g0[:, 0].numpy(), g0[:, 1].numpy()
    cat = np.full(S, "tilted", dtype=object)
    cat[gz < -0.7] = "upright"
    cat[gz > 0.7] = "inverted"
    cat[(np.abs(gx) > 0.7) | (np.abs(gy) > 0.7)] = "side"

    def pct(a: np.ndarray) -> dict:
        return {f"p{q}": round(float(np.percentile(a, q)), 3) for q in (10, 50, 90)}

    diag = {k: np.concatenate([d[k] for d in diag_all]) for k in diag_all[0]}
    res = {
        "checkpoint": args.checkpoint,
        "num_envs": n, "episodes": args.episodes, "samples": S,
        "A_upstream_criterion": {
            "success_last_second": float(strict_all.float().mean()),
            "success_per_step_mean": float(step_all.float().mean()),
            "recovery_step_median": float(torch.nanmedian(rev_a[strict_all])) if strict_all.any() else None,
        },
        "B_leg_joints_only": {
            "success_last_second": float(strict_leg.float().mean()),
            "success_per_step_mean": float(step_leg.float().mean()),
            "recovery_step_median": float(torch.nanmedian(rev_b[strict_leg])) if strict_leg.any() else None,
        },
        "C_paper_flat_terrain_criterion": {
            "success_last_second": float(strict_paper.float().mean()),
            "success_per_step_mean": float(step_paper.float().mean()),
            "recovery_step_median": float(torch.nanmedian(rev_c[strict_paper])) if strict_paper.any() else None,
        },
        "stand_up_height_and_orientation_at_end": float(stand.float().mean()),
        "cond_pass_rate_last_second": {
            CONDS[i]: float(tail[:, :, i].float().mean()) for i in range(len(CONDS))
        },
        "cond_pass_rate_at_last_step": {
            CONDS[i]: float(conds[:, -1, i].float().mean()) for i in range(len(CONDS))
        },
        "final_distributions": {
            "base_height_m": pct(diag["height"]),
            "orientation_err": pct(diag["ori_err"]),
            "leg_joint_dev_rad": pct(diag["jdev_leg"]),
            "all_joint_dev_rad": pct(diag["jdev_all"]),
            "leg_max_joint_vel_rad_s": pct(diag["vmax_leg"]),
            "all_max_joint_vel_rad_s": pct(diag["vmax_all"]),
        },
        "by_initial_pose": {},
    }
    for c in ["upright", "inverted", "side", "tilted"]:
        m = torch.from_numpy(cat == c)
        if m.any():
            res["by_initial_pose"][c] = {
                "samples": int(m.sum()),
                "A_success": float(strict_all[m].float().mean()),
                "B_success": float(strict_leg[m].float().mean()),
                "C_success": float(strict_paper[m].float().mean()),
                "stand_at_end": float(stand[m].float().mean()),
            }

    A, B, C = res["A_upstream_criterion"], res["B_leg_joints_only"], res["C_paper_flat_terrain_criterion"]
    print("\n============ 评估结果（最后 1 秒持续达标；S = 样本数）============", flush=True)
    print(f"检查点                  : {os.path.basename(args.checkpoint)}", flush=True)
    print(f"样本数                  : {S}  ({n} envs × {args.episodes} episodes)", flush=True)
    print(f"A) 上游判据(含轮关节)   : {A['success_last_second']*100:.2f}%", flush=True)
    print(f"B) 修正判据(只看腿关节) : {B['success_last_second']*100:.2f}%", flush=True)
    print(f"C) 论文判据(高度0.42+腿): {C['success_last_second']*100:.2f}%   ← 与论文 97.8%/99.1% 同口径", flush=True)
    print(f"高度+姿态达标(最后一步) : {res['stand_up_height_and_orientation_at_end']*100:.2f}%", flush=True)
    if C["recovery_step_median"] is not None:
        print(f"C 口径恢复耗时中位数    : {C['recovery_step_median']:.0f} 步 / 50 Hz = "
              f"{C['recovery_step_median']/50:.2f} s", flush=True)
    print("\n单项通过率（最后 1 秒）:", flush=True)
    for k, v in res["cond_pass_rate_last_second"].items():
        print(f"   {k:<8} {v*100:6.1f}%", flush=True)
    print("\n末态分布 (p10 / p50 / p90):", flush=True)
    for k, v in res["final_distributions"].items():
        print(f"   {k:<26} {v['p10']:8.3f} {v['p50']:8.3f} {v['p90']:8.3f}", flush=True)
    print("\n按初始倒地姿态:", flush=True)
    for k, v in res["by_initial_pose"].items():
        print(f"   {k:<9} 样本 {v['samples']:5d}  A {v['A_success']*100:6.2f}%  "
              f"B {v['B_success']*100:6.2f}%  C {v['C_success']*100:6.2f}%  "
              f"站姿 {v['stand_at_end']*100:6.2f}%", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n结果已存: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    import sys as _sys
    _sys.stdout.flush()
    os._exit(code)
