# 本地修改说明（本分支 = 上游 + 论文标准口径）

本分支由 `uglycatcat/petbot2_rl` 维护，内容 = 上游 `Kitjesen/recovery`（commit `5b574775`）
+ 我们为**按论文复现**所做的修改（第三代"论文标准口径"，与实验 `p1_paper_exact` 同源）。

`default.xml` 里 recovery 项目的 `remote` 指向 `ours`、`revision` 指向本分支，`repo sync` 可直接得到这份修改后的版本。

## 修改内容

| 文件 | 改动 | 原因 |
|---|---|---|
| `scripts/train.py` | 去掉不存在的 `isaaclab.utils.io.dump_pickle` 导入，用标准库 `pickle` 补同名函数 | 该 API 在**任何已发布**的 Isaac Lab 中都不存在，否则 `train.py` 直接 ImportError；补丁语义与原用法一致（仅影响 `params/*.pkl` 落盘） |
| `src/thunder_recovery/mdp/_utils.py` | **`_get_cw` 改为论文 Eq.3 的递增语义** `CW(i) = β^(decay^i)`（0.3 → 1）；`_get_ed` 用 Eq.1 原式 `((a·t)/T)^k`（a = T/2）；保留 `_get_ed_norm` 供早阶段门控 | 上游 `_get_cw` 是**衰减到 0**（iter 1000 为 2.3e−15），行为惩罚被提前关掉，导致"站得住但还在晃"；上游 `_get_ed` 归一化后起身阶段任务奖励弱约 15 倍 |
| `src/thunder_recovery/mdp/method1_deng/rewards.py` | 奖励组合按 Eq.4（行为组只乘 CW、不含 DS）；行为惩罚权重恢复 Table I；姿态项改二次 `(g_b−e_z)²`；删除论文没有的 `wheel_leg_coord` 与 `joint_deviation`；`support_state` 去掉自加的高度门 | 对齐论文原文 |
| `src/thunder_recovery/mdp/noisy_actions.py`（新增） | 位置 / 速度动作项的高斯动作噪声（σ = 0.02） | 论文 §IV-A 的 DR 项，Isaac Lab 原动作项无此接口 |
| `src/thunder_recovery/config/method1_deng/env_cfg.py` | episode 改回 **5 s @ 50 Hz**（§IV-A 原文）；动作缩放 0.5；DR 按 §IV-A（基座 ±5 kg、连杆 ±10%、摩擦 U[0.7,1.3]、刚度/阻尼 ±1.0/±0.1）；接入动作噪声 | 同上 |

完整 diff（相对上游 commit，含新增文件）：`patches/paper_standard.patch`。
第一轮（尚未纠正 CW 与 episode 长度的版本）留档：`patches/paper_repro_v1.patch`。

## 与实验树的差异

本分支与训练服务器上的 `p1_paper_exact` 实验树**行为完全一致**，唯一差异是两处**注释 / docstring 更正**
（`_get_cw` 的说明、`env_cfg.__post_init__` 的 episode 说明）—— 上游遗留的旧说明与实际实现相反，容易误导，故在此更正。

## 逐实验的差异

各轮实验（论文口径的多个变体、Go2-W 版本、CW 语义对照等）**不**放在本分支，而是作为补丁保存于
分支 `refproj/recovery-tools` 的 `tools/recovery_trial/variant_diffs/`（每个实验一个完整 diff）。
