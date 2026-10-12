# 论文 vs 上游实现（Kitjesen/recovery）：参数对照与偏差清单

对照来源：

- 论文：Deng et al., *Learning to Recover: Dynamic Reward Shaping with Wheel-Leg Coordination for
  Fallen Robots*, arXiv:2506.05516 —— §III-A/III-C/III-D/III-E、Table I、Eq.(1)(3)(4)、§IV-A。
- 参考实现（作者的部署栈）：`../Recovery_go2w/simulate_python/test/runtest.py`（含 `model_7999.pt`
  的 Actor 定义、obs 组装、动作缩放、PD 增益、50 Hz 策略频率）。
- 被评估实现：`../recovery/src/thunder_recovery/**`（Thunder 平台）。

## 1. 训练/环境基本量

| 项 | 论文 | 上游实现 | 影响 |
|---|---|---|---|
| episode 长度 | **4 s** | **5 s**（`episode_length_s=5.0`） | horizon 不同 |
| 控制频率 | **100 Hz**（→400 控制步） | **50 Hz**（`decimation 4 × dt 0.005`，→250 步） | 控制步数差 1.6 倍；论文的 DS 标定假设 4 s |
| 自由落体 | 1.1 m、零力矩、2 s | 1.1 m、零力矩、2 s（`FREEFALL_STEPS=100` @50 Hz） | ✅ 一致 |
| 终止 | 仅 timeout | 仅 timeout | ✅ 一致 |
| PPO | lr 1e-3、γ=0.99、asymmetric actor-critic | lr 1e-3、γ=0.99、asymmetric | ✅ 一致 |
| 训练规模 | 未明写；评测 2048 个随机初态 | 4096 envs × 10000 iter（=19.7 亿样本） | 规模不是瓶颈（本轮已跑满） |

## 2. 动态奖励整形 DS / 课程 CW —— **主要偏差所在**

| 项 | 论文 | 上游实现 | 影响 |
|---|---|---|---|
| DS 公式 | `DS = ((a·t)/T)^k`，**a = T/2**，k = 3 → 对 4 s episode：`DS(t) = (t/2)³`，**自由落体结束（t=2 s）时 DS=1.0，末尾 8.0** | `ED = (t/T)³ ∈ [0,1]`（`_utils.py:_get_ed`），t=2 s 时 **ED≈0.064**，末尾 1.0 | 起身阶段（t=2 s 起）任务奖励**弱约 15 倍**，末尾**弱 8 倍**；论文靠"后段精确姿态收敛"取胜，这一项被严重稀释 |
| CW | 0.3 起、0.968 指数衰减 | `0.3 × 0.968^i` | ✅ 一致 |
| DS 作用范围 | Eq.(4)：DS 乘 joint_pos、action_rate、joint_vel、torque、joint_acc；**高度/姿态/轮速不受 DS 缩放** | ED 乘 stand_joint_pos、base_height、base_orientation（+wheel_velocity） | 后段行为项权重结构不同 |
| CW 作用范围 | Eq.(4)：CW 乘整组行为项（collision + action_rate + joint_vel + torque + joint_acc） | CW 只乘 collision + action_rate | 后段平滑/能耗约束弱很多 |

## 3. 奖励项权重（论文 Table I vs 实现）

| 项 | 论文 | 实现 | 备注 |
|---|---|---|---|
| stand joint position | 42 | 42（`exp(−Σ(q*−q)²/(σ²N))`，σ=0.5） | ✅ |
| base height | 120（h*=**0.42**） | 120（h*=**0.426**，σ=0.1） | ✅ 近似 |
| base orientation | 50，**`(g_b − e_z)²` 二次惩罚** | 50，**`exp(−|g_b − e_z|²)` 有界奖励** | ❗形态不同：倒置时论文代价可达 ~200，实现只把奖励压到 0 → 对"从完全倒置纠正姿态"的推力弱得多 |
| body collision | −5e−2 | **−1e−2**（缩小 5×） | 实现注释称有意缩小 |
| action rate | −1e−2（排除轮动作） | **−2e−3**（仅腿，缩小 5×） | ❗ |
| joint velocity | −2e−2 | **−3e−3**（缩小 ~7×） | ❗ 论文靠此项压住"末段速度<0.1" |
| torques | −2.5e−5 | **−5e−6**（缩小 5×） | ❗ |
| acceleration | −2.5e−7 | **−5e−8**（缩小 5×） | ❗ |
| wheel velocity | −2e−2 | **−3e−3**（缩小 ~7×） | ❗ |
| 额外项（论文无） | — | `support_state` +5、`wheel_leg_coord` +5、`joint_deviation` −2e−3 | 实现自己的解释 |

## 4. 动作与执行器

| 项 | 论文/作者栈 | 上游实现 | 影响 |
|---|---|---|---|
| 腿位置动作缩放 | 作者部署代码：`a*0.5 + q_default`（s_p = 0.5） | **hip 0.125 / thigh·calf 0.25** | ❗探索幅度只有作者的一半～四分之一（init_noise_std=1.0 下 ±1σ 只能动 0.125–0.25 rad），翻身所需的大幅摆动被压缩 |
| 轮速度动作缩放 | 作者部署：0.5（s_v<1.0） | 0.8 | 论文只说 s_v<1.0 |
| 腿伺服增益 | 作者部署 Kp=25 / Kd=0.5 | DCMotor **stiffness 100 / damping 5** | 本项目仿真伺服比作者部署硬 4 倍（论文训练期增益未公开，只给了 DR 扰动 ±15/±2） |
| 轮伺服 | Kd_wheel=0.1 | ImplicitActuator damping 1 | 相位差 |

## 5. 观测（actor 78 维）

| 项 | 论文/作者栈 | 上游实现 | 结论 |
|---|---|---|---|
| 维度与顺序 | 16 动作 + 3 投影重力 + 3 角速度 + 12 腿位置 + 12 腿速度 + 4 轮速 + 12/12/4 上一帧 | 完全相同（`env_cfg.py` PolicyCfg 9 项） | ✅ 结构对齐（这也是能直接加载作者 actor 权重的依据） |
| base 线速度 | 论文正文说 actor 观测它（带高噪声）；**作者代码里没有** | 没有 | 以作者代码为准，✅ |
| critic 维度 | 作者：103 | 实现：285（height_scan 等） | 推理无关，仅训练期 |

## 6. 成功判据

| 项 | 论文 | 实现 | 影响 |
|---|---|---|---|
| 基座高度 | **> 0.42 m** | > **0.30 m** | 实现更宽松 |
| 关节偏差 | ≤ 0.5 rad | < 0.5 rad，**对全部 16 个关节（含 4 个 continuous 轮关节）**求范数 | ❗轮角无限累积 → 该项近乎永不满足 |
| 最大关节速度 | < 0.1 rad/s | < 0.1 rad/s，同样含轮关节 | 同上 |
| 姿态误差 | 平地实验的判定未含此项；非平地实验用 < 0.1 | < 0.1，且要求**最后 1 秒持续满足** | 实现更严的时间要求 |
| 采样量 | 2048 个随机初态 | 我们评估用 6144 样本 | ✅ 可比 |

## 7. 域随机化

| 项 | 论文 §IV-A | 上游实现 | 结论 |
|---|---|---|---|
| 摩擦 | 静摩擦 U[**0.7, 1.3**] | 静摩擦 U[**0.3, 1.0**]，动摩擦 U[0.3,0.8]，restitution U[0,0.5] | 区间不同且整体更低 |
| 基座质量 | **±5.0 kg** / episode | 基座 scale (0.9, 1.1) = ±10% | 量纲不同（Thunder 整机轻，±10% 可能 <5 kg） |
| 连杆质量 | ±10% | 未做 | ❗缺 |
| 执行器刚度/阻尼 | **±15 / ±2** 扰动 | 未做 | ❗缺 |
| 动作噪声 | **σ=0.02** | 未做（仅观测噪声） | ❗缺 |
| 其他 | — | 额外加了 reset 外扰 ±10 N/±10 N·m、10–15 s 周期推力 | 实现自加 |

## 8. 结论（偏差排序）

复现失败（真实成功率 0%、末态高度中位数仅 0.203 m、侧躺 13%、倒置 0%）最可能的原因是下面这几条**与论文不一致的设计选择**，按可疑度排序：

1. **腿动作缩放只有作者的 1/2～1/4**（0.125/0.25 vs 0.5）—— 直接压缩翻身所需的动作幅度与探索范围。
2. **DS/ED 标定偏了**：论文 `a=T/2` 让 DS 在自由落体结束时就等于 1、末尾到 8；实现把它归一化到 [0,1]，起身阶段任务奖励弱约 15 倍。
3. **行为惩罚权重被整体下调 5～7 倍**，且没有按论文 Eq.(4) 用 CW 覆盖整组行为项 —— 论文靠这些项把末段关节速度压到 <0.1 rad/s。
4. **base orientation 用有界 exp 替代论文的二次惩罚** —— 从完全倒置姿态回正的梯度推力弱得多。
5. **缺 3 项 DR**（连杆质量 ±10%、执行器刚度/阻尼 ±15/±2、动作噪声 σ=0.02），摩擦区间也和论文不同。
6. **成功判据把 continuous 轮关节算进角度/速度**（指标本身失真，但不影响策略行为）。

反向证据：论文自己的 baseline（不含 DS）也有 96.4%，说明奖励结构细节不是全部原因 —— 上述偏差叠加足以让论文指标无法复现。

作者权重可用于诊断的边界：`model_7999.pt` 的 actor 与我们的 actor **张量形状完全一致**（78→16、[128,128,128]、std=1.0），所以技术上能加载；但它是 **Go2-W** 平台训练的权重，论文的跨平台结论是"方法可迁移"（两平台分别训练），**不是权重可迁移**，且关节顺序需按 FL/FR/RL/RR 重排、默认姿态与腿长都不同 —— 因此"用作者权重跑我们的 Thunder 环境"得到的结果无法区分是实现问题还是跨平台差异，不适合作为判据。

## 9. 已按论文口径修改（repro v1，2026-10-10）

补丁存档：`paper_repro_v1.patch`（相对上游 `5b574775`，含 train.py 的 `dump_pickle` 补丁；共 4 文件 +96/−72 行）。
训练实验名：`paper_v1`（4096 envs × 10000 iter，seed 42）。

| # | 改动 | 位置 |
|---|---|---|
| 1 | 腿位置动作缩放 → **0.5**（原 hip 0.125 / thigh·calf 0.25）；轮动作缩放 0.8 → **0.5** | `env_cfg.py` ActionsCfg |
| 2 | DS 改为论文 Eq.1 原式 **`((a·t)/T)^k`，a = T/2**（原为归一化 `(t/T)^k`）；另留 `_get_ed_norm` 供早阶段门控 | `mdp/_utils.py` |
| 3 | 行为惩罚权重恢复 Table I：collision −5e−2、action_rate −1e−2、joint_vel −2e−2、torque −2.5e−5、accel −2.5e−7、wheel_vel −2e−2 | `env_cfg.py` RewardsCfg |
| 4 | base orientation 改为论文的二次惩罚 **(g_b − e_z)²，权重 −50**（原为有界 exp） | `rewards.py` + `env_cfg.py` |
| 5 | 按 Eq.4 重排 DS/CW：joint_pos ×DS；height/ori/wheel_vel 不乘 DS；行为组 ×CW（其中 action_rate/joint_vel/torque/accel 再 ×DS） | `rewards.py` |
| 6 | 移除上游自加的 `recovery_joint_deviation` 惩罚（不在论文中） | `env_cfg.py` |
| 7 | 域随机化按 §IV-A：摩擦 U[0.7,1.3]（原 [0.3,1.0]）、基座质量 **±5 kg**(add)、新增连杆质量 ±10%、新增执行器刚度 ±15/阻尼 ±2；移除论文没有的周期推力与 reset 外扰 | `env_cfg.py` EventsCfg |
| 8 | episode **4 s**（原 5 s），保持 50 Hz 控制（与作者部署栈策略周期一致） | `env_cfg.py` `__post_init__` |

保留的上游自有设计（论文未规定，已注明）：`support_state` 带 0.30 m 高度门、`wheel_leg_coord`（早阶段 `1-ED_norm` 门控）、body collision 的力裁剪、free-fall 期间的项屏蔽。
**尚未实现**：论文的动作噪声 σ=0.02（Isaac Lab 动作项没有对应接口，需要自定义 action term）。
评估口径：`eval_recovery.py` 现同时给出 A（上游原样）、B（只看腿关节）、**C（论文平地判据：高度>0.42 + 腿关节）**，C 与论文 97.8%/99.1% 直接可比。

## 10. 建议的下一步（按性价比）

1. **参数消融阶梯（推荐，先做）**：在保持其它不变的前提下，先只改①腿动作缩放 0.5、②DS 改回 `a=T/2`，各跑 500–1000 iter（每次约 1 小时），看末态高度/姿态是否显著改善；再逐项加回论文的行为权重与二次姿态项。这样能定位到底哪一项在起作用，而不是盲跑 12 小时。
2. **论文忠实复现跑**：按上表把所有偏差改齐（4 s/100 Hz、DS 常数、Table I 权重、二次姿态项、DR 三项、动作缩放 0.5），跑满 10000 iter（约 12 h），用同一评估入口判定。
3. 若要验证作者权重本身，应在**作者的 MuJoCo 栈**里做（`Recovery_go2w/simulate_python/unitree_mujoco.py` + `test/runtest.py` + 自带 `unitree_robots/go2w`），需要装 unitree_sdk2py 并跑他们的 DDS 桥，属于另一条独立工作线。
