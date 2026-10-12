# 本地修改说明（本分支 = 上游 + 我们的补丁）

本分支由 `uglycatcat/petbot2_rl` 维护，内容 = 上游 `Kitjesen/recovery`（commit `5b574775`）+ 我们为**试训复现论文**所做的修改。
`default.xml` 里 recovery 项目的 `remote` 指向 `ours`、`revision` 指向本分支，`repo sync` 可直接得到这份修改后的版本。

## 修改内容

| 文件 | 改动 | 原因 |
|---|---|---|
| `scripts/train.py` | 去掉不存在的 `isaaclab.utils.io.dump_pickle` 导入，改用标准库 `pickle` 补一个同名函数 | Isaac Lab 2.3.2（及所有已发布版本）没有该 API，否则 `train.py` 直接 ImportError；补丁语义与原用法一致（仅影响 `params/*.pkl` 落盘） |
| `src/thunder_recovery/config/method1_deng/env_cfg.py` | 按论文口径改：腿动作缩放 0.5、episode 5 s、行为惩罚权重用论文 Table I、姿态项用 `(g_b−e_z)²`、按 Eq.4 重排 DS/CW、DR 按 §IV-A（含动作噪声 σ=0.02）、去掉论文没有的项 | 对齐论文原文，见 `patches/paper_repro_v1.patch` |
| `src/thunder_recovery/mdp/_utils.py` | DS 用论文 Eq.1 `((a·t)/T)^k`（a = T/2）；CW 用论文 Eq.3 `cw ← cw^β`（0.3 递增到 1） | 上游实现为"归一化 ED + 衰减 CW"，与论文语义相反 |
| `src/thunder_recovery/mdp/method1_deng/rewards.py` | 奖励组合与单项形式按论文 Table I / Eq.4 校正 | 同上 |

完整 diff：`patches/paper_repro_v1.patch`、`patches/upstream_dump_pickle.patch`。
变更依据与实测记录：`uglycatcat/petbot2_rl` 的 `docs/reference_recovery_validation.md`。

上游其余内容未改动；训练产物（`logs/`）不进本分支。
