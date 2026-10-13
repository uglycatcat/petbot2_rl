#!/usr/bin/env python3
"""诊断：作者 Go2-W 权重的复刻是否自洽。

1) 保持默认关节姿态（a=0）时的稳态基座高度 → 判定“论文 0.42 m”判据对该 MJCF 是否合理
2) 从站立姿态出发跑策略 → 观测是否给对（正确的话策略应能维持站姿、动作很小）
3) 倒地后跑策略，逐 0.5 s 打印高度/倾角/动作幅值 → 看它到底有没有起身趋势
"""
import numpy as np
import torch

import mujoco
from go2w_policy_check import Go2W, make_policy, KP, KD, KD_WHEEL, CTRL_EVERY, SIM_DT, SDK_FROM_POL, ACTION_SCALE

SCENE = "/workspace/variants/g1_go2w_paper/assets/go2w/scene.xml"
CKPT = "/workspace/variants/g1_go2w_paper/assets/model_7999.pt"


def hold(bot, a, seconds, q_target=None):
    """用 a（策略输出格式）驱动 seconds 秒；a=None 时直接保持默认姿态。"""
    n = int(seconds / (SIM_DT * CTRL_EVERY))
    for _ in range(n):
        if q_target is None:
            q_t = a[:12] * ACTION_SCALE + bot.default_leg
            v_t = a[12:] * ACTION_SCALE
        else:
            q_t, v_t = q_target, np.zeros(4)
        for _ in range(CTRL_EVERY):
            tau_l = np.clip(KP * (q_t - bot.leg_pos()) - KD * bot.leg_vel(), -23.7, 23.7)
            tau_w = np.clip(KD_WHEEL * (v_t - bot.wheel_vel()), -15, 15)
            bot.data.ctrl[bot.leg_act] = tau_l
            bot.data.ctrl[bot.whl_act] = tau_w
            mujoco.mj_step(bot.model, bot.data)


def set_state(bot, z=0.5, quat=(1, 0, 0, 0), leg=None):
    mujoco.mj_resetData(bot.model, bot.data)
    bot.data.qpos[0:3] = [0, 0, z]
    bot.data.qpos[3:7] = quat
    bot.data.qpos[bot.leg_qadr] = bot.default_leg if leg is None else leg
    bot.data.qvel[:] = 0
    mujoco.mj_forward(bot.model, bot.data)


def policy_act(bot, pol, pre_actions, prev):
    obs = np.concatenate([pre_actions, bot.gravity_body(), bot.angvel_body(),
                          bot.leg_pos(), bot.leg_vel(), bot.wheel_vel(), *prev])
    with torch.no_grad():
        a = pol(torch.tensor(obs, dtype=torch.float32)).numpy()
    targets16 = np.concatenate([a[:12] * ACTION_SCALE + bot.default_leg, a[12:] * ACTION_SCALE])
    return a, targets16[SDK_FROM_POL]


bot = Go2W(SCENE)
pol = make_policy(CKPT)
print("策略关节顺序:", bot.pol_legs, bot.pol_wheels)
print("默认腿关节角:", bot.default_leg)

# ── 1) 保持默认姿态的稳态高度 ──
set_state(bot, z=0.45)
hold(bot, None, 2.0, q_target=bot.default_leg)
print(f"\n[1] 保持默认姿态 2 s 后的基座高度 = {bot.base_height():.3f} m "
      f"(腿偏差 {np.abs(bot.leg_pos()-bot.default_leg).max():.3f})")

# ── 2) 站立姿态出发，跑策略 5 s ──
set_state(bot, z=bot.base_height())
pre = np.concatenate([bot.default_leg, np.zeros(4)])[SDK_FROM_POL]
prev = (bot.leg_pos(), bot.leg_vel(), bot.wheel_vel())
trace = []
for k in range(int(5.0 / (SIM_DT * CTRL_EVERY))):
    a, pre = policy_act(bot, pol, pre, prev)
    prev = (bot.leg_pos(), bot.leg_vel(), bot.wheel_vel())
    hold(bot, a, SIM_DT * CTRL_EVERY)
    if k % 25 == 0:
        trace.append((k * 0.02, bot.base_height(), float(np.abs(a).max()), float(np.abs(a[:12]).mean())))
print("\n[2] 从站姿出发（正确的话应稳住）: t / 高度 / max|a| / mean|a_leg|")
for row in trace:
    print("    t=%4.1fs  z=%.3f  max|a|=%.2f  mean|a_leg|=%.2f" % row)

# ── 3) 倒地 → 策略，逐 0.5 s 打印 ──
rng = np.random.default_rng(7)
bot.reset_fallen(rng)
print(f"\n[3] 落地瞬间 z={bot.base_height():.3f}，倾角 "
      f"{np.degrees(np.arccos(np.clip(-bot.gravity_body()[2],-1,1))):.0f}°")
pre = np.concatenate([bot.default_leg, np.zeros(4)])[SDK_FROM_POL]
prev = (bot.leg_pos(), bot.leg_vel(), bot.wheel_vel())
for k in range(int(5.0 / (SIM_DT * CTRL_EVERY))):
    a, pre = policy_act(bot, pol, pre, prev)
    prev = (bot.leg_pos(), bot.leg_vel(), bot.wheel_vel())
    hold(bot, a, SIM_DT * CTRL_EVERY)
    if k % 25 == 0:
        tilt = np.degrees(np.arccos(np.clip(-bot.gravity_body()[2], -1, 1)))
        print("    t=%4.1fs  z=%.3f  倾角=%3.0f°  max|a|=%.2f  腿偏差=%.2f" % (
            k * 0.02, bot.base_height(), tilt, float(np.abs(a).max()),
            float(np.abs(bot.leg_pos() - bot.default_leg).max())))
