# PetBot 自恢复 RL

PetBot（轮足机器狗）在平地上被推倒、滚翻或跌落后，通过强化学习策略恢复到默认站立构型并保持稳定。

- **当前阶段：试跑参考项目。** 正在训练服务器上试跑成熟的开源参考项目（当前是 `recovery`），**先挑出最合适的参考项目**，再开始 PetBot 自身的训练。
- **与参考项目的本质区别**：PetBot 本来就能轻松站住；而轮足翻倒起身类参考项目默认连"站住"都还没解决 —— 它们的奖励权重、课程与成功判据不能直接照搬。
- 主线：Isaac Lab + RSL-RL + PPO 训练 → 策略导出 → MuJoCo 做 sim2sim 验证。
- **进度、全局规则与设备判断见 [AGENTS.md](AGENTS.md)（agent 请先读它）；文档索引见 [docs/README.md](docs/README.md)。**
- 目录职责：

| 路径 | 内容 |
|---|---|
| [robot_description/](robot_description/README.md) | 机器人模型（MJCF + STL）。模型、物理参数与网格**禁止修改** |
| [docs/](docs/README.md) | 需求基线、工程实施、参考项目试训记录 |
| `reference_project/` | 上游参考项目，由根目录 [default.xml](default.xml) 清单管理；不进版本管理 |
| `robot_controller_demo/` | 早期手动键盘 demo，**已废弃**，无需阅读或参考 |

## 如何配置

前置：`git`、`repo`（Ubuntu 可 `apt install repo`，或把[官方 launcher](https://gerrit.googlesource.com/git-repo/) 放进 `~/.local/bin`：`curl -o ~/.local/bin/repo https://storage.googleapis.com/git-repo-downloads/repo && chmod a+rx ~/.local/bin/repo`）、能访问 GitHub 的网络。

```bash
# 1. 取得本仓库（可放在任意目录）
git clone git@github.com:uglycatcat/petbot2_rl.git petbot2_rl   # 或 https://github.com/uglycatcat/petbot2_rl.git
cd petbot2_rl

# 2. 用清单物化参考项目（检出到 reference_project/<项目名>/）
#    -b main 必须带：本仓库默认分支是 main，不带它 repo 会去找 master 并报
#    "couldn't find remote ref refs/heads/master"
repo init -b main -u git@github.com:uglycatcat/petbot2_rl.git -m default.xml
repo sync                 # 只要某类项目：repo sync -g recovery

# 3. 以后更新参考项目：
repo sync                 # 按 default.xml 里锁定的 revision 对齐
```

**网络受限环境的两个额外参数（训练服务器实测）**：

- `repo init` 默认从 github.com 拉取 repo 工具本体（HTTPS）—— 训练服务器上该方向**超时**，会一直卡住；加镜像源即可：
  `--repo-url https://mirrors.tuna.tsinghua.edu.cn/git/git-repo`（TUNA 与 `https://gh-proxy.com/https://github.com/GerritCodeReview/git-repo` 均实测可用）。
- `repo` 用清单里的 `fetch` 拼出的是 **HTTPS URL**，所以在只有 SSH 能通的网络里，`repo sync` 同步上游项目还要把 `https://github.com/` 重写为 `git@github.com:`（例如 `git config --global url."git@github.com:".insteadOf "https://github.com/"`，会改动全局 git 配置，执行前先确认）。
- **已验证（2026-10-12，训练服务器）**：`repo init -b main --repo-url <TUNA> -m default.xml` + `repo manifest` 解析通过 —— 10 个项目，`reference_project/recovery` 解析为 `remote=ours, revision=refproj/recovery`（即 `uglycatcat/petbot2_rl` 的该分支）。**未执行 `repo sync`**（会下载 8 个未物化的上游项目，属额外授权范围）。

- [`default.xml`](default.xml) 是参考项目的**唯一清单**：新增、删除或改变某个项目的 pin，必须在同一次改动里同步更新它和 [`reference_project/manifest.json`](reference_project/manifest.json)。
- 参考项目默认浅克隆并锁在确定的 commit 上；它们**不进本仓库的版本管理**（见 [.gitignore](.gitignore)）。
- 上游的**本地修改**与我方**试训工具**分别保存在分支 `refproj/recovery` 与 `refproj/recovery-tools`（后者内容在 `tools/recovery_trial/`）；不要只留在被忽略的目录里。
- 训练与实验在**训练服务器**上进行；桌面机不训练，只在最后做效果验证（两台设备的判断方法见 [AGENTS.md](AGENTS.md)）。
- 参考项目的用途、来源与使用边界见 [reference_project/README.md](reference_project/README.md)。
