# PetBot RL 项目

本仓库管理 PetBot 轮足机器人的强化学习实验、MuJoCo 演示与参考项目。功能策略按起立、低台阶、跌倒恢复三个独立 demo 管理；任务范围见[需求基线](docs/current_scope.md)，代码入口见[demo 导览](demo_project/README.md)。

- 主线：Isaac Lab + RSL-RL + PPO 训练 → 策略导出 → MuJoCo 做 sim2sim 验证。
- **Agent 请先读 [AGENTS.md](AGENTS.md)**；当前交接见[当前交接页](docs/current_handoff.md)，文档按[索引](docs/README.md)选读。本 README 只负责配置与目录导览。
- 目录职责：

| 路径 | 内容 |
|---|---|
| [robot_description/](robot_description/README.md) | 机器人模型（MJCF + STL）。模型、物理参数与网格**禁止修改** |
| [docs/](docs/README.md) | 当前交接、需求、工程约定与实验记录 |
| `reference_project/` | 上游参考项目，由根目录 [default.xml](default.xml) 清单管理；不进版本管理 |
| [demo_project/](demo_project/README.md) | 三个功能 demo 与历史实验；同一 repo 清单管理，代码在 `demoproj/` 分支，见[管理约定](docs/engineering_handoff.md#版本管理与双端协作) |
| [robot_controller_demo/](robot_controller_demo/README.md) | 本机 MuJoCo 固定姿态恢复与被动沉降统计。早期键盘驾驶只作历史记录 |

## 如何配置

前置：`git`、`repo`（Ubuntu 可直接 `apt install repo`，或把[官方 launcher](https://gerrit.googlesource.com/git-repo/) 放进 `~/.local/bin`）、能访问 GitHub 的网络。

```bash
# 1. 取得本仓库（可放在任意目录）
git clone https://github.com/uglycatcat/petbot2_rl.git petbot2_rl
cd petbot2_rl

# 2. 用清单物化参考项目与 demo（分别检出到 reference_project/、demo_project/）
repo init -u https://github.com/uglycatcat/petbot2_rl.git -b main -m default.xml
repo sync                 # 只要某类项目：repo sync -g recovery

# 3. 以后按清单更新嵌套项目（先保全各 checkout 的本地改动）：
repo sync                 # 按 default.xml 里锁定的 revision 对齐
```

- [`default.xml`](default.xml) 是参考项目与 demo 的**统一清单**：参考项目条目变化同时更新 [`reference_project/manifest.json`](reference_project/manifest.json)；demo 条目变化同时更新 [demo manifest](demo_project/manifest.json) 和索引。开发 checkout 保留完整历史，demo 代码推送到对应 `demoproj/` 分支。
- 参考项目默认浅克隆并锁在确定的 commit 上；它们**不进本仓库的版本管理**（见 [.gitignore](.gitignore)）。
- 训练与实验在**训练服务器**上进行；桌面机不训练，只在最后做效果验证（两台设备的判断方法见 [AGENTS.md](AGENTS.md)）。
- 参考项目的用途、来源与使用边界见 [reference_project/README.md](reference_project/README.md)。
