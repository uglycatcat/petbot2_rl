# 当前交接：停止旧训练，按需求独立重建upright

更新：2026-10-30。先读AGENTS.md，再读本页。旧入口、旧实验完成事件和历史自动授权均不覆盖本次暂停。

## 最新指令与执行边界

用户要求停止正在进行的训练并重整项目，进一步明确：继承式训练之后的代码不可作为复用基础，后续澄清的需求仍有效。当前只保留需求和只读机器人模型作为重建依据，不再以旧奖励、课程、网络、控制周期或scratch入口为默认方案。

**训练、工程短训、评估发队列和自动迭代暂停。** 不恢复第26批，不加载历史模型；无周期定时器或goal。旧实现与实验记录封存追溯，现有桌面包未改动，不是新工程能力证明。恢复训练须新明确指令。

## 已完成的停止与封存

- 第26批两路训练及启动器、完成/失败观察器均已退出。训练退出143是用户取消，无显存/PhysX/Python中断；请求停止时日志约454/457更新，各保存至model_400.pt，未完成1000更新，未做终局评估。
- 控制目录`demo_project/runs/_upright_cycle_batch26`已将launch_status/eval_status/eval_finished结算为cancelled_by_user，并保存取消前状态、cancellation和completion_event_reconciliation。不得当失败自动恢复，也不得当完成或能力通过。
- 旧demo源码b357dd7已推送归档分支`archive/upright_rl_demo_pre_rebuild_20261030`，保留全部Git历史。完整本地目录保存在`demo_project/runs/_upright_reorganization_20261030/checkout_before_rebuild`；180个跟踪文件哈希一致，生成物及未跟踪文件也保留。
- 以前的需求/交接分别原样保存为[current_scope_archive_20261030.md](current_scope_archive_20261030.md)、[current_handoff_archive_20261030.md](current_handoff_archive_20261030.md)。历史实验文档和固定运行副本未删除。

## 当前工程实际状态

`demo_project/upright_rl_demo`已与旧实现隔离，当前仅有需求规格、重建设计、只读规格检查和明确拒绝启动的训练预留入口。**训练算法、仿真环境和完整课程尚未重建，没有新版能力结果。** 不把结构整理、XML检查或旧体验写成从零训练成功。

入口见[demo README](../demo_project/upright_rl_demo/README.md)、[需求基线](current_scope.md)、[重建计划](../demo_project/upright_rl_demo/docs/rebuild_plan.md)、[架构边界](../demo_project/upright_rl_demo/docs/architecture.md)。重整验收见[upright_reorganization_20261030.md](upright_reorganization_20261030.md)。

## 后续工作顺序

先按需求独立设计模型/控制接口、部署观测和单次调用状态，再建立从零静止起立与真实支撑学习门槛；同次训练内扩展保持、落地和完整循环，之后探索共同髋初态、双向连续0–0.75m/s及镜像联合运动。具体网络、控制周期、奖励、课程及判据尚待实现和验证，不能复用归档代码替代。

所有用户目标保持：普通四轮全能力独立、Space直线起立且少调整、两轮共同髋自主平衡、落地只增默认髋位且继承速度和交接、意外退出不自动重起、±25rad/s目标及源模型只读、每份正式1000更新、ONNX后做真实MuJoCo闭环。正式训练目录保持空，用户亲自体验认可后才讨论迁移。

## 历史资料的使用边界

历史第十几批完整流程与C58桌面体验作为事实保留，但不再作为新实现的复用或优化依据。第24/25批失败、v28工程结果均只留档；第26批为未完成的取消实验。旧移动第五轮及以后废弃结果仍不恢复。历史编号见[谱系](upright_training_lineage_20261023.md)，不需要全读docs或runs。
