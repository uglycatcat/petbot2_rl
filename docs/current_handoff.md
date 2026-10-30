# 当前交接：upright第4轮低速遥控通过，第5轮分阶段扩能力已启动

更新：2026-10-31。先读AGENTS.md，再读本页。upright按最新GPU2/3从零训练授权执行，旧第26批不恢复；完成事件先核查实际状态再分析迭代。

## 最新指令与执行边界

用户持续授权单Actor–Critic从零训练、评估和事件驱动自动迭代，仅GPU2/3。第4轮2训练/6独立评估有效，无显存中断或恢复；两种子默认起立、静止循环、30秒低速完整和基本遥控均各32/32。单种子60秒完整32/32、基本31/32，落地加速/偏航与实测轮速超调仍在，不能宣称全部能力或MuJoCo通过。见[终局](upright_restart_round04_results.md)。第5轮固定6ee0adc，分离两轮速度/接管速度/髋初态、按实测晋级、保留20%较低难度回归，独立验证60秒；工程门槛后两种子各2048环境×1000更新已随机初始化启动。见[执行](upright_restart_round05_execution.md)。无历史权重、周期定时器/goal。

用户最新纠正：需求与期望效果已明确，早期已实现基本起立、两轮与落地，应核查并复用有效代码和经验，重点是训练代码迭代效率，不必全部重新设计。此前将整个旧实现永久排除的解释范围过大，已纠正；完整封存仍保留；已选择性复用静止v5资产/分布代码并实现独立完整循环。新训练仍禁止历史权重和冻结父依赖。

**此前仅修改的暂停已被最新自动迭代授权替代。** 不恢复第26批、不加载历史模型；既有C58桌面包未改动，仅作体验基准，不作新工程能力证明。

## 已完成的停止与封存

- 第26批两路训练及启动器、完成/失败观察器均已退出。训练退出143是用户取消，无显存/PhysX/Python中断；请求停止时日志约454/457更新，各保存至model_400.pt，未完成1000更新，未做终局评估。
- 控制目录`demo_project/runs/_upright_cycle_batch26`已将launch_status/eval_status/eval_finished结算为cancelled_by_user，并保存取消前状态、cancellation和completion_event_reconciliation。不得当失败自动恢复，也不得当完成或能力通过。
- 旧demo源码b357dd7已推送归档分支`archive/upright_rl_demo_pre_rebuild_20261030`，保留全部Git历史。完整本地目录保存在`demo_project/runs/_upright_reorganization_20261030/checkout_before_rebuild`；180个跟踪文件哈希一致，生成物及未跟踪文件也保留。
- 以前的需求/交接分别原样保存为[current_scope_archive_20261030.md](current_scope_archive_20261030.md)、[current_handoff_archive_20261030.md](current_handoff_archive_20261030.md)。历史实验文档和固定运行副本未删除。

## 当前工程实际状态

`demo_project/upright_rl_demo`当前配置configs/round05.json，49项容器CPU、128环境8更新及4→8同次恢复、2048环境2更新、32回合独立重载有效。高阶段场景物理检查显式使用第4轮固定网络，属于工程开发验证，不是新PPO初始化或独立终局。正式和短训PPO均随机初始化；起落控制、奖励、源物理不变。两轮训练30秒、独立验证60秒；课程0/1每50更新检查，运动每100更新，连续两次至少80%晋级，终局不晋级到未训练阶段。后续分析必须单列落地加速、偏航、实际轮速超调和真实速度响应。

入口见[demo README](../demo_project/upright_rl_demo/README.md)、[需求基线](current_scope.md)、[重建计划](../demo_project/upright_rl_demo/docs/rebuild_plan.md)、[架构边界](../demo_project/upright_rl_demo/docs/architecture.md)。重整验收见[upright_reorganization_20261030.md](upright_reorganization_20261030.md)。

## 后续工作顺序

**新编号：** 系列upright_restart_20261030，当前upright_restart_round05；控制目录`demo_project/runs/_upright_restart_round05`，运行`upright_restart_round05_s101`/`upright_restart_round05_s102`。状态与实际进程为准，运行中无eval_finished正常。第4轮终局状态、全部192轨迹、随机初始化和开发失败审计齐全，原进程已退出，不重复启动。补充三项能力验证位于runs/upright_restart_round04_capability，已有效结束。

用户最新明确：需求已经清晰，历史实现的筛选和设计由agent判断；优先以较短、多批次demo训练快速验证实现。1000更新不是最终交付上限，正式从零流程可按收敛和验收需要使用10000或更多更新。本次为预算与职责澄清，未重新启动已取消队列。

本批完成后先读launch_status/eval_status/eval_finished、实际进程、缺失/失败与显存中断，区分开发课程检查和独立验证。起立→静止循环→60秒低速；已训练更高课程且低速基本至少26/32时再做该阶段60秒验证，其余显式跳过。读取请求/执行/实测与成功/失败，尤其过渡动作质量；按结果修改并过工程门槛后再从零迭代。完成/异常事件已接回当前会话，退出观察器为一次性PID等待。

所有用户目标保持：普通四轮全能力独立、Space直线起立且少调整、两轮共同髋自主平衡、落地只增默认髋位且继承速度和交接、意外退出不自动重起、±25rad/s目标及源模型只读、当前demo单份1000更新、最终交付预算按收敛配置且可达10000或更多、ONNX后做真实MuJoCo闭环。正式训练目录保持空，用户亲自体验认可后才讨论迁移。

## 历史资料的使用边界

历史第十几批完整流程与C58桌面体验作为事实保留，早期有效源码可核查后复用。实际谱系显示完整循环第1批已有actor迁移、第12批已有冻结父网络、第17批继承第16批；不能简单以20批作为依赖分界。静止v5有独立从零训练证据，可用于候选来源审查；选定基础前仍需核查代码与模型适用性。第24/25批失败、v28工程结果均只留档；第26批为未完成的取消实验。旧移动第五轮及以后废弃结果仍不恢复。历史编号见[谱系](upright_training_lineage_20261023.md)，不需要全读docs或runs。
