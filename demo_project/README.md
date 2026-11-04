# PetBot 训练项目导览

这里存放 PetBot 自有 demo，独立于 `reference_project/` 的上游项目。根目录 [`default.xml`](../default.xml) 管理嵌套 checkout，[`manifest.json`](manifest.json) 记录来源、分支与已核对提交；main 只跟踪索引和清单，demo 源码在各自 `demoproj/<项目名>` 分支。

| 项目 | 职责 | 当前开发状态 |
|---|---|---|
| [upright_rl_demo](upright_rl_demo/README.md) | 辅助存档，保持当前源码版本。 | upright已冻结；当前有效21r5 seed102训练代码在[1104工程](../rl_project/train_upright_project_1104/README.md)，状态见[冻结说明](../docs/upright_1104.md)。 |
| [step_rl_demo](step_rl_demo/README.md) | 固定2cm/30cm、同步髋轮；单次抬轮、归髋净行15cm后停稳。 | 用户授权v2第1批完整从零训练，固定`cdd6c95`、GPU0/1两路512×500；能力待终局评估，见[执行](../docs/step_v2_round01_20261104.md)。 |
| [recovery_rl_demo](recovery_rl_demo/README.md) | 重新设计分阶段跌倒恢复。 | 仅登记与需求说明，尚无训练代码。 |
| [recovery_agile_01](recovery_agile_01/PETBOT.md) | 基于AGILE的早期恢复原型。 | 历史实验，不能直接作为新恢复任务的设计或成功基线。 |
| [jump_rl_demo](jump_rl_demo/README.md) | 跳跃可行性预研。 | 完整跳跃已放弃，保留历史入口。 |

## `runs/` 与源码的区别

`runs/` 保存当前保留的demo批次的**固定提交副本**、模型检查点、TensorBoard 日志、评估轨迹、状态 JSON 和诊断产物。它累积了多轮、多种子及多次评估，所以目录和体积远大于 `upright_rl_demo/`。正式实验从固定副本运行，防止后续源代码修改改变在途实验；这些副本与产物不在 main 跟踪。不要凭目录数量推断当前并发数，也不要为了整理目录而移动或删除运行中的产物。当前批次只从[交接页](../docs/current_handoff.md)给出的状态文件定位。

`upright_rl_demo/`按用户要求原样保留作辅助存档，不是当前有效版本入口。1104完整训练源码另存于`rl_project/train_upright_project_1104/`；sim2sim继续使用21r5 seed102。全部upright runs已迁出到服务器history，其他任务的runs仍按原流程管理；源`robot_description/`只读。

存档内README的“当前候选/训练入口”和旧执行开关只反映当时状态，不构成执行授权；upright工程短训、固定策略评估及自动迭代均冻结。旧文档链接按[历史定位](../docs/upright_history.md)查找，当前接手以[1104说明](../docs/upright_1104.md)为准。

## 同步与开发

- 开发前检查嵌套仓库分支与远程，干净工作区同步相应 `demoproj/` 分支；两端同一 demo 只由一端写入。主仓库 `git pull` 不会自动更新嵌套 checkout。
- 训练先固定 commit，再建立独立运行副本；在途副本和检查点不热改。每个完整源码修改及时提交并推送对应 demo 分支，更新 `manifest.json` 的已核对提交；新增、删除或改变分支时，同步 `default.xml` 和索引。
- 训练仅在专用容器内，源模型只读。正式发批前完成任务契约、GPU资源和检查点重载验收；按多批次迭代，批次划分、每批训练数量、时长、GPU分配与计算利用率由agent按评估结果和资源情况安排，不设固定利用率目标。整卡显存守护78%，避让其他用户。
- 用户已取消周期定时续接；既有队列是否续接按当前交接与最新用户授权处理，勿重复启动。运行状态和下一步见[当前交接](../docs/current_handoff.md)，实验细节从[文档索引](../docs/README.md)按需查阅。

全部upright系列文档与运行产物已迁入服务器`history/upright/`，不进Git；按[归档说明](../docs/upright_history.md)检索，日常仅阅读[1104冻结说明](../docs/upright_1104.md)。
