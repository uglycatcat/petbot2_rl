#!/usr/bin/env python3
"""用作者发布的 Go2-W 权重（model_7999.pt）在 MuJoCo 上直接验证论文结论。

严格复刻作者部署栈（Recovery_go2w/simulate_python/test/runtest.py）的推理契约：
  obs(78) = pre_actions(16) + projected_gravity(3) + ang_vel_body(3) + 腿位(12)
            + 腿速(12) + 轮速(4) + 上一帧腿位(12) + 上一帧腿速(12) + 上一帧轮速(4)
  关节顺序：按关节类型分组（全部 hip → thigh → calf），腿序 FL→FR→RL→RR
            （= runtest 的置换 [3,0,9,6,4,1,10,7,5,2,11,8]；轮 [13,12,15,14]）
  动作：q_target = a*0.5 + q_default（腿）、v_target = a*0.5（轮）
        腿 τ = Kp(q_t−q) − Kd·qd（Kp 25 / Kd 0.5）；轮 τ = Kd_w(v_t−v)（Kd_w 0.1）
        力矩上限取 MJCF ctrlrange（hip/thigh ±23.7、calf ±45.43、wheel ±15）
  控制 50 Hz（sim dt 0.002 × 10 步），episode 5 s。

场景（论文 §III-A）：随机基座姿态 + 随机关节角 → 零力矩自由落体 1.1 m / 2 s → 策略接管
判据（论文 §IV-B）：末步高度 > 0.42 m、12 腿关节相对默认姿态偏差 ≤ 0.5 rad、最大关节速度 < 0.1 rad/s
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

import mujoco

DEFAULT_LEG_VALS = (0.0, 0.8, -1.5)     # hip / thigh / calf（unitree_legged_const.default_pos）
KP, KD, KD_WHEEL = 25.0, 0.5, 0.1
ACTION_SCALE = 0.5
SIM_DT = 0.002
CTRL_EVERY = 10                          # → 50 Hz
EPISODE_S = 5.0
FREEFALL_S = 2.0
DROP_H = 1.1
LEGS = ("FL", "FR", "RL", "RR")
LEG_TYPES = ("hip", "thigh", "calf")
# 策略输出顺序：hip(FL,FR,RL,RR) → thigh(…) → calf(…) → wheel(FL,FR,RL,RR)
# SDK/电机顺序：按腿分组 FR,FL,RR,RL × hip,thigh,calf，再接 FR,FL,RR,RL 的轮
# （= runtest.py 的置换 [1,5,9,0,4,8,3,7,11,2,6,10,13,12,15,14]）
POL_NAMES = [f"{leg}_{t}" for t in LEG_TYPES for leg in LEGS] + [f"{leg}_wheel" for leg in LEGS]
SDK_NAMES = [f"{leg}_{t}" for leg in ("FR", "FL", "RR", "RL") for t in LEG_TYPES] \
            + [f"{leg}_wheel" for leg in ("FR", "FL", "RR", "RL")]
SDK_FROM_POL = [POL_NAMES.index(n) for n in SDK_NAMES]


class Actor(nn.Module):
    """与 runtest.load_policy 完全一致的结构（78→16, [128,128,128]）。"""

    def __init__(self, input_dim: int = 78, act_dim: int = 16):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim),
        )
        self.critic = nn.Sequential(nn.Linear(103, 128), nn.ReLU(), nn.Linear(128, 128),
                                    nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.std = nn.Parameter(torch.ones(act_dim))

    def forward(self, obs: torch.Tensor):
        return self.actor(obs)


def make_policy(ckpt: str) -> Actor:
    pol = Actor()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"]
    pol.load_state_dict(state)
    pol.eval()
    return pol


class Go2W:
    """MJCF 与策略关节顺序之间的映射 + PD/速度控制。"""

    def __init__(self, scene_xml: str):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

        # 策略顺序：hip(FL,FR,RL,RR) → thigh(…) → calf(…)；轮：FL,FR,RL,RR
        self.pol_legs = [f"{leg}_{t}" for t in LEG_TYPES for leg in LEGS]
        self.pol_wheels = [f"{leg}_wheel" for leg in LEGS]
        self.leg_qadr = np.array([m.jnt_qposadr[jid(n + "_joint")] for n in self.pol_legs])
        self.leg_dadr = np.array([m.jnt_dofadr[jid(n + "_joint")] for n in self.pol_legs])
        self.whl_dadr = np.array([m.jnt_dofadr[jid(n + "_joint")] for n in self.pol_wheels])
        self.leg_act = np.array([aid(n) for n in self.pol_legs])
        self.whl_act = np.array([aid(n) for n in self.pol_wheels])
        self.lim_leg = np.array([m.actuator_ctrlrange[i][1] for i in self.leg_act])
        self.lim_whl = np.array([m.actuator_ctrlrange[i][1] for i in self.whl_act])
        self.base_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.default_leg = np.array([DEFAULT_LEG_VALS[LEG_TYPES.index(t)] for t in LEG_TYPES for _ in LEGS])

    # ── 状态 ──
    def leg_pos(self):      return self.data.qpos[self.leg_qadr].copy()
    def leg_vel(self):      return self.data.qvel[self.leg_dadr].copy()
    def wheel_vel(self):    return self.data.qvel[self.whl_dadr].copy()
    def base_height(self):  return float(self.data.qpos[2])
    def _rot(self):         return self.data.xmat[self.base_body].reshape(3, 3)
    def gravity_body(self): return self._rot().T @ np.array([0.0, 0.0, -1.0])
    def angvel_body(self):  return self._rot().T @ self.data.qvel[3:6]

    def reset_fallen(self, rng: np.random.Generator, jitter: float = 0.6):
        mujoco.mj_resetData(self.model, self.data)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)                        # 均匀随机姿态（MuJoCo 用 w,x,y,z）
        self.data.qpos[0:3] = [0.0, 0.0, DROP_H]
        self.data.qpos[3:7] = q
        self.data.qpos[self.leg_qadr] = self.default_leg + rng.uniform(-jitter, jitter, 12)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(FREEFALL_S / SIM_DT)):      # 零力矩自由落体
            self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)

    def step_control(self, a: np.ndarray):
        q_t = a[:12] * ACTION_SCALE + self.default_leg
        v_t = a[12:] * ACTION_SCALE
        for _ in range(CTRL_EVERY):
            tau_leg = np.clip(KP * (q_t - self.leg_pos()) - KD * self.leg_vel(), -self.lim_leg, self.lim_leg)
            tau_whl = np.clip(KD_WHEEL * (v_t - self.wheel_vel()), -self.lim_whl, self.lim_whl)
            self.data.ctrl[self.leg_act] = tau_leg
            self.data.ctrl[self.whl_act] = tau_whl
            mujoco.mj_step(self.model, self.data)


STANCE_H = {"h": None}       # 平台自身站姿高度，运行期测出


def measure_stance(bot: Go2W) -> float:
    """保持默认关节姿态 2 s 后的稳态基座高度（该平台的实际站姿高度）。"""
    mujoco.mj_resetData(bot.model, bot.data)
    bot.data.qpos[0:3] = [0, 0, 0.45]
    bot.data.qpos[3:7] = [1, 0, 0, 0]
    bot.data.qpos[bot.leg_qadr] = bot.default_leg
    bot.data.qvel[:] = 0.0
    mujoco.mj_forward(bot.model, bot.data)
    for _ in range(int(2.0 / SIM_DT)):
        tau_l = np.clip(KP * (bot.default_leg - bot.leg_pos()) - KD * bot.leg_vel(), -23.7, 23.7)
        tau_w = np.clip(KD_WHEEL * (0.0 - bot.wheel_vel()), -15, 15)
        bot.data.ctrl[bot.leg_act] = tau_l
        bot.data.ctrl[bot.whl_act] = tau_w
        mujoco.mj_step(bot.model, bot.data)
    return bot.base_height()


def rollout(pol: Actor, bot: Go2W, rng: np.random.Generator, renderer=None, frames=None):
    bot.reset_fallen(rng)
    tilt = math.degrees(math.acos(np.clip(-bot.gravity_body()[2], -1.0, 1.0)))

    prev_leg, prev_leg_vel, prev_whl_vel = bot.leg_pos(), bot.leg_vel(), bot.wheel_vel()
    pre_actions = np.zeros(16)
    vel_hist = []
    n_ctrl = int(EPISODE_S / (SIM_DT * CTRL_EVERY))
    for k in range(n_ctrl):
        obs = np.concatenate([pre_actions, bot.gravity_body(), bot.angvel_body(),
                              bot.leg_pos(), bot.leg_vel(), bot.wheel_vel(),
                              prev_leg, prev_leg_vel, prev_whl_vel])
        with torch.no_grad():
            a = pol(torch.tensor(obs, dtype=torch.float32)).numpy()
        # 作者实现：pre_actions = (a*0.5 + q_default) 的电机顺序排列
        targets16 = np.concatenate([a[:12] * ACTION_SCALE + bot.default_leg,
                                    a[12:] * ACTION_SCALE])
        pre_actions = targets16[SDK_FROM_POL]
        prev_leg, prev_leg_vel, prev_whl_vel = bot.leg_pos(), bot.leg_vel(), bot.wheel_vel()
        bot.step_control(a)
        vel_hist.append(float(np.abs(bot.leg_vel()).max()))
        if renderer is not None and frames is not None and k % 2 == 0:
            renderer.update_scene(bot.data, camera=-1)
            frames.append(renderer.render())

    dev = np.abs(bot.leg_pos() - bot.default_leg)
    last1s = int(1.0 / (SIM_DT * CTRL_EVERY))
    m = dict(
        height=bot.base_height(),
        leg_dev_max=float(dev.max()),
        max_leg_vel_last1s=float(np.max(vel_hist[-last1s:])),
        max_leg_vel_final=float(np.abs(bot.leg_vel()).max()),
        init_tilt_deg=float(tilt),
    )
    sh = STANCE_H["h"] or 0.42
    m["final_tilt_deg"] = float(np.degrees(np.arccos(np.clip(-bot.gravity_body()[2], -1, 1))))
    m["success"] = bool(m["height"] > 0.42 and m["leg_dev_max"] <= 0.5 and m["max_leg_vel_last1s"] < 0.1)
    # 平台自适应判据：站到自身站姿高度的 92% 以上且姿态接近竖直、关节偏差与速度达标
    m["success_platform"] = bool(m["height"] > 0.92 * sh and m["final_tilt_deg"] < 20.0
                                 and m["leg_dev_max"] <= 0.5 and m["max_leg_vel_last1s"] < 0.1)
    return m


def bucket(deg: float) -> str:
    if deg < 45:
        return "upright"
    if deg > 135:
        return "inverted"
    return "side"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="/workspace/variants/g1_go2w_paper/assets/go2w/scene.xml")
    ap.add_argument("--checkpoint", default="/workspace/variants/g1_go2w_paper/assets/model_7999.pt")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", type=int, default=0, help="额外渲染 N 个 episode 的 mp4")
    ap.add_argument("--out", default="/workspace/variants/g1_go2w_paper/logs/go2w_policy_check.json")
    args = ap.parse_args()

    pol = make_policy(args.checkpoint)
    bot = Go2W(args.scene)
    STANCE_H["h"] = measure_stance(bot)
    print(f"该平台自身站姿高度 = {STANCE_H['h']:.3f} m（论文 0.42 m 阈值不适用于 Go2-W）")
    rng = np.random.default_rng(args.seed)

    rows = [rollout(pol, bot, rng) for _ in range(args.episodes)]

    n = len(rows)
    succ = sum(r["success"] for r in rows)
    print("\n===== 作者官方 Go2-W 权重（model_7999.pt）在论文机器人上的验证 =====")
    print(f"样本数              : {n}（随机倒地 → 自由落体 {DROP_H} m / {FREEFALL_S} s 后策略接管 {EPISODE_S} s）")
    sp = sum(r["success_platform"] for r in rows)
    print(f"论文原判据(>0.42m)     : {succ / n * 100:.2f}%   ← 0.42 m 高于该平台站姿高度，恒不达标")
    print(f"平台自适应判据成功率  : {sp / n * 100:.2f}%   （高度>92%站姿 + 倾角<20° + 腿偏差≤0.5 + 速度<0.1）")
    for key, zh in (("final_tilt_deg", "末步倾角(度)"), ("height", "末步基座高度(m)"), ("leg_dev_max", "腿关节偏差 max(rad)"),
                    ("max_leg_vel_last1s", "最后1s最大关节速度(rad/s)")):
        v = np.array([r[key] for r in rows])
        print(f"{zh:<20}: p10 {np.percentile(v,10):7.3f}  p50 {np.percentile(v,50):7.3f}  p90 {np.percentile(v,90):7.3f}")
    print("按落地姿态分层（落地瞬间基座与竖直夹角）：")
    for b in ("upright", "side", "inverted"):
        sub = [r for r in rows if bucket(r["init_tilt_deg"]) == b]
        if sub:
            print(f"  {b:<9} n={len(sub):<4} 成功率 {sum(x['success'] for x in sub)/len(sub)*100:6.2f}%"
                  f"   末高 p50 {np.percentile([x['height'] for x in sub],50):.3f} m")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n": n, "success_rate": succ / n, "rows": rows}, fh, indent=1)
    print(f"\n明细已存: {args.out}")

    if args.video:
        import imageio
        r = mujoco.Renderer(bot.model, height=480, width=640)
        for i in range(args.video):
            frames = []
            rollout(pol, bot, rng, renderer=r, frames=frames)
            path = f"/workspace/variants/g1_go2w_paper/logs/go2w_policy_ep{i}.mp4"
            imageio.mimsave(path, frames, fps=25)
            print("视频:", path)
        r.close()


if __name__ == "__main__":
    main()
