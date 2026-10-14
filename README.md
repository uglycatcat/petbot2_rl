# PetBot 自恢复 RL

PetBot（轮足机器狗）在平地上被推倒、滚翻或跌落后，通过强化学习策略恢复到默认站立构型并保持稳定。

- **当前阶段：参考试训与 PetBot 首版实现并行准备。** recovery 线已关闭。AGILE 的**标准起身训练已于 2026-10-13 在服务器容器内启动**（`StandUp-T1-v0`、4096 envs、100k iteration；实测 ≈7.4 s/iteration → 约 8.8 天），robot_lab 容器就绪未开训。已确定以 AGILE 为底座、借鉴 robot_lab 混合动作配置实现 `demo_project/recovery_agile_01/`，无需等待参考试训结束。PetBot 首版代码已在本地实现并备份（容器 `petbot-train` 就绪），9 项 CPU 测试通过，尚未运行 PhysX 或训练；见 [demo 索引](demo_project/README.md)。
- **与参考项目的本质区别**：PetBot 本来就能轻松站住；而轮足翻倒起身类参考项目默认连"站住"都还没解决 —— 它们的奖励权重、课程与成功判据不能直接照搬。
- 主线：Isaac Lab + RSL-RL + PPO 训练 → 策略导出 → MuJoCo 做 sim2sim 验证。
- **进度、全局规则与设备判断见 [AGENTS.md](AGENTS.md)（agent 请先读它）；文档索引见 [docs/README.md](docs/README.md)。**
- 目录职责：

| 路径 | 内容 |
|---|---|
| [robot_description/](robot_description/README.md) | 机器人模型（MJCF + STL）。模型、物理参数与网格**禁止修改** |
| [docs/](docs/README.md) | 需求基线、工程实施、参考项目试训记录 |
| `reference_project/` | 上游参考项目，由根目录 [default.xml](default.xml) 清单管理；不进版本管理 |
| [demo_project/](demo_project/README.md) | PetBot 实验项目，首个为 `recovery_agile_01`；同一 repo 清单管理，代码备份到 `demoproj/` 分支，见 [实施约定](docs/engineering_handoff.md#版本管理与双端协作) |
| [robot_controller_demo/](robot_controller_demo/README.md) | 本机 MuJoCo 固定姿态恢复与被动沉降统计。早期键盘驾驶只作历史记录 |

## 如何配置

前置：`git`、`repo`（Ubuntu 可直接 `apt install repo`，或把[官方 launcher](https://gerrit.googlesource.com/git-repo/) 放进 `~/.local/bin`）、能访问 GitHub 的网络。

```bash
# 1. 取得本仓库（可放在任意目录）
git clone https://github.com/uglycatcat/petbot2_rl.git petbot2_rl
cd petbot2_rl

# 2. 用清单物化参考项目（检出到 reference_project/<项目名>/）
repo init -u https://github.com/uglycatcat/petbot2_rl.git -b main -m default.xml
repo sync                 # 只要某类项目：repo sync -g recovery

# 3. 以后更新参考项目：
repo sync                 # 按 default.xml 里锁定的 revision 对齐
```

- [`default.xml`](default.xml) 是参考项目与 demo 的**统一清单**：新增、删除或改变某个项目的 pin，必须在同一次改动里同步更新它和 [`reference_project/manifest.json`](reference_project/manifest.json)。
- demo 的新增、删除和分支变化同步记录在 [demo manifest](demo_project/manifest.json)；开发 checkout 保留完整历史，代码推送到 `demoproj/` 分支。
- 参考项目默认浅克隆并锁在确定的 commit 上；它们**不进本仓库的版本管理**（见 [.gitignore](.gitignore)）。
- 训练与实验在**训练服务器**上进行；桌面机不训练，只在最后做效果验证（两台设备的判断方法见 [AGENTS.md](AGENTS.md)）。
- 参考项目的用途、来源与使用边界见 [reference_project/README.md](reference_project/README.md)。
