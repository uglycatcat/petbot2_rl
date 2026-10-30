# PetBot 训练项目导览

这里存放 PetBot 自有 demo，独立于 `reference_project/` 的上游项目。根目录 [`default.xml`](../default.xml) 管理嵌套 checkout，[`manifest.json`](manifest.json) 记录来源、分支与已核对提交；main 只跟踪索引和清单，demo 源码在各自 `demoproj/<项目名>` 分支。

| 项目 | 职责 | 当前开发状态 |
|---|---|---|
| [upright_rl_demo](upright_rl_demo/README.md) | 按用户需求独立重建从随机初始化到完整起落与两轮遥控的工程。 | 已停训；旧实现完整封存，当前只有需求、设计和只读检查，训练尚未重建。见[当前交接](../docs/current_handoff.md)。 |
| [step_rl_demo](step_rl_demo/README.md) | 前轮登上0–3 cm低台阶后加速爬升。 | 仅登记与需求说明，尚无训练代码。 |
| [recovery_rl_demo](recovery_rl_demo/README.md) | 重新设计分阶段跌倒恢复。 | 仅登记与需求说明，尚无训练代码。 |
| [recovery_agile_01](recovery_agile_01/PETBOT.md) | 基于AGILE的早期恢复原型。 | 历史实验，不能直接作为新恢复任务的设计或成功基线。 |
| [jump_rl_demo](jump_rl_demo/README.md) | 跳跃可行性预研。 | 完整跳跃已放弃，保留历史入口。 |

## `runs/` 与源码的区别

`runs/` 保存所有 demo 各批次的**固定提交副本**、模型检查点、TensorBoard 日志、评估轨迹、状态 JSON 和诊断产物。它累积了多轮、多种子及多次评估，所以目录和体积远大于 `upright_rl_demo/`。正式实验从固定副本运行，防止后续源代码修改改变在途实验；这些副本与产物不在 main 跟踪。不要凭目录数量推断当前并发数，也不要为了整理目录而移动或删除运行中的产物。当前批次只从[交接页](../docs/current_handoff.md)给出的状态文件定位。

`upright_rl_demo/` 是**可维护源码**，当前入口为`train_scratch.py`，并保留历史任务评估实现；它不只保存最新代码，但分支 HEAD 只表示目前可维护版本。upright继续在此迭代，直到用户亲自确认sim2sim可以通过后才讨论正式训练迁移；演示与验证入口在`rl_project/std_s2s_upright_project/`。旧检查点仍须配套其固定提交、任务契约和运行配置使用。生成的 `artifacts/` 不等于源模型，`robot_description/` 只读。

## 同步与开发

- 开发前检查嵌套仓库分支与远程，干净工作区同步相应 `demoproj/` 分支；两端同一 demo 只由一端写入。主仓库 `git pull` 不会自动更新嵌套 checkout。
- 训练先固定 commit，再建立独立运行副本；在途副本和检查点不热改。每个完整源码修改及时提交并推送对应 demo 分支，更新 `manifest.json` 的已核对提交；新增、删除或改变分支时，同步 `default.xml` 和索引。
- 训练仅在专用容器内，源模型只读。正式发批前完成任务契约、GPU资源和检查点重载验收；按多批次迭代，批次划分、每批训练数量、时长、GPU分配与计算利用率由agent按评估结果和资源情况安排，不设固定利用率目标。整卡显存守护78%，避让其他用户。
- 用户已取消周期定时续接；既有队列是否续接按当前交接与最新用户授权处理，勿重复启动。运行状态和下一步见[当前交接](../docs/current_handoff.md)，实验细节从[文档索引](../docs/README.md)按需查阅。
