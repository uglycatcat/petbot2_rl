# PetBot 实验训练项目

本目录放我们自己的训练项目，独立于 `reference_project/` 的参考实验。根目录 [default.xml](../default.xml) 统一管理 checkout；[manifest.json](manifest.json) 记录来源、备份分支和已核对提交。main 只管理索引、清单和忽略规则；嵌套代码在本仓库 `demoproj/<项目名>` 分支版本化。

| 项目 | 底座 | 分支 | 状态 |
|---|---|---|---|
| [recovery_agile_01](recovery_agile_01/PETBOT.md) | WBC-AGILE `6830cf9`，借鉴 robot_lab 混合动作 | `demoproj/recovery_agile_01` | 服务器链路跑通；首轮 11 组 2048×3000 扫描 + 确定性独立评估完成：**无一组学到起身**（最好 5 组 0.225~0.227 与开环地板基线 0.213 持平）；见[首版扫描记录](recovery_agile_01/PETBOT_SWEEP_01.md) |

## 同步与开发纪律

- 已初始化 repo 的工作区，更新主线清单后可用 `repo sync demo_project/recovery_agile_01` 物化该项目；repo 的 manifest checkout 与主仓库 main 是两份 Git 状态，主仓库 `git pull` 本身不会更新所有嵌套 checkout。
- 同一 demo 同时只由一端修改；另一端运行固定 commit 的独立副本。禁止同步正在训练的代码副本。
- 修改前检查、同步远程分支；未提交/未推送的改动先保全，分叉先合并。repo 管理检出，提交与推送仍需在 demo checkout 中执行，不能在 detached HEAD 上遗留提交。
- 每个完整修改及时 commit、push；阶段结束不得只留本机副本。首次分支推送成功后再写清单，新增/删除/改变来源或分支时同步更新本索引、manifest.json 和 default.xml。
- **我方工具/脚本**（容器定义与挂载约定、并行扫描调度、确定性评估、地板基线）备份在 `demoproj/<项目名>-tools` 分支，内容放 `tools/petbot_trial/`，不进 `default.xml`。注意 demo 仓库带 git-lfs，**基于 demo 历史开新分支推送需要上传全部 LFS 对象**，所以工具分支做成**独立孤儿分支**（只含 `tools/`）；取用方式：`git fetch ours demoproj/<项目名>-tools` 后把 `tools/petbot_trial/` 拷到 `demo_project/demo_env/`。
- 开发用完整 Git 历史；日志、检查点、生成资产与初态库不直接提交 Git，记录代码 commit、模型内容指纹、配置和产物位置。

在 repo 创建的 checkout 里，先检查 `git remote -v`；清单 remote 名为 `ours`，本机首次 clone 的 remote 名为 `origin`，不要假定两者相同。首次开发可用 `git switch -c demoproj/recovery_agile_01 --track ours/demoproj/recovery_agile_01` 建立跟踪分支；若已有同名分支，切回已有分支，不要覆盖。干净工作区内 `git pull --ff-only` 后开始修改，分叉则保全并处理。

当前桌面 checkout 是独立 clone；清单已登记，后续由 repo 按同一路径物化。完整控制契约、容器内命令与验证边界见项目 [PETBOT.md](recovery_agile_01/PETBOT.md)。2026-10-13 在训练服务器 `petbot-train` 容器内跑通资产转换、PhysX 导入、任务实例化、pre-learn 与冒烟训练（证据与两处适配见 PETBOT.md 的「服务器验证记录」）。

## 首轮结果复核与解释边界（2026-10-14）

15:03 在训练服务器核对磁盘产物：11 个变体均保存了 `model_2999.pt`，每组确定性评估为 512 episodes，结果与首轮汇总一致。`base_s42` 为 116/512（22.65625%），同批初态的开环默认构型基线为 109/512（21.28906%），零动作基线为 0/512。尚不能据此宣称已学到有效的动态起身能力。

**解释边界**：21.3% 是持续发送默认构型指令后的成功率，不能解释为“21% 初态已满足成功判据”或机器人完全不运动。扫描记录同时报告初态构型条件通过率为 0；因此仅排除初始已成功样本，未必能排除这些开环可恢复样本。奖励信号、探索噪声与初态分布是待进一步对照验证的原因，当前数据不充分排除训练预算或其它因素。嵌套 demo 的 PETBOT.md / PETBOT_SWEEP_01.md 中相关强因果表述应按此边界理解。

AGILE 同时在继续训练：日志 `reference_project/agile_env/logs/agile_standup_20261013_1742.log` 到 10774/100000，iteration time 7.47 s，日志 ETA 约 7 天 7 小时；最新检查点为 `reference_project/WBC-AGILE/logs/rsl_rl/stand_up_t1/2026-10-13_09-42-56_stand_up_t1/model_10750.pt`。这些是训练进度，不能替代独立恢复成功率评估。
