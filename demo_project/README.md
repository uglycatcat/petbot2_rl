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
- 开发用完整 Git 历史；日志、检查点、生成资产与初态库不直接提交 Git，记录代码 commit、模型内容指纹、配置和产物位置。

在 repo 创建的 checkout 里，先检查 `git remote -v`；清单 remote 名为 `ours`，本机首次 clone 的 remote 名为 `origin`，不要假定两者相同。首次开发可用 `git switch -c demoproj/recovery_agile_01 --track ours/demoproj/recovery_agile_01` 建立跟踪分支；若已有同名分支，切回已有分支，不要覆盖。干净工作区内 `git pull --ff-only` 后开始修改，分叉则保全并处理。

当前桌面 checkout 是独立 clone；清单已登记，后续由 repo 按同一路径物化。完整控制契约、容器内命令与验证边界见项目 [PETBOT.md](recovery_agile_01/PETBOT.md)。2026-10-13 在训练服务器 `petbot-train` 容器内跑通资产转换、PhysX 导入、任务实例化、pre-learn 与冒烟训练（证据与两处适配见 PETBOT.md 的「服务器验证记录」）。
