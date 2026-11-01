# 当前交接：upright第18轮已验收，第19轮大误差成本从零训练已启动

更新：2026-11-02。先读AGENTS.md，再读本页。upright按最新GPU2/3从零训练授权执行，旧第26批不恢复；完成事件先核查实际状态再分析迭代。

## 最新指令与执行边界

用户持续授权单Actor–Critic从零训练、评估和事件迭代，仅GPU2/3。第18轮两份1000更新与8独立评估有效，无缺失/OOM/恢复；低速完整/基本均64/64，高范围完整/基本均60/64，四次两轮前轮回落。实测前.455–.483/后.524–.551m/s，轮速峰值47.089rad/s，全部能力未通过。第19轮固定3a5680a仅补最大.25奖励/秒、尺度.30m/s的大误差成本，总速度成本上界1.75<支撑2/秒；保留精细奖励、Beta512、actor195/critic219和全部控制/评分。109项CPU及GPU短训恢复/重载/默认规模有效，固定旧actor物理/评分不变、奖励差逐帧核对，GPU2/3各2048×1000已从零进入PPO。见[第18轮终局](upright_restart_round18_results.md)、[第19轮执行](upright_restart_round19_execution.md)。无历史训练权重、周期定时器/goal。

用户最新纠正：需求与期望效果已明确，早期已实现基本起立、两轮与落地，应核查并复用有效代码和经验，重点是训练代码迭代效率，不必全部重新设计。此前将整个旧实现永久排除的解释范围过大，已纠正；完整封存仍保留；已选择性复用静止v5资产/分布代码并实现独立完整循环。新训练仍禁止历史权重和冻结父依赖。

**此前仅修改的暂停已被最新自动迭代授权替代。** 不恢复第26批、不加载历史模型；既有C58桌面包未改动，仅作体验基准，不作新工程能力证明。

## 已完成的停止与封存

- 第26批两路训练及启动器、完成/失败观察器均已退出。训练退出143是用户取消，无显存/PhysX/Python中断；请求停止时日志约454/457更新，各保存至model_400.pt，未完成1000更新，未做终局评估。
- 控制目录`demo_project/runs/_upright_cycle_batch26`已将launch_status/eval_status/eval_finished结算为cancelled_by_user，并保存取消前状态、cancellation和completion_event_reconciliation。不得当失败自动恢复，也不得当完成或能力通过。
- 旧demo源码b357dd7已推送归档分支`archive/upright_rl_demo_pre_rebuild_20261030`，保留全部Git历史。完整本地目录保存在`demo_project/runs/_upright_reorganization_20261030/checkout_before_rebuild`；180个跟踪文件哈希一致，生成物及未跟踪文件也保留。
- 以前的需求/交接分别原样保存为[current_scope_archive_20261030.md](current_scope_archive_20261030.md)、[current_handoff_archive_20261030.md](current_handoff_archive_20261030.md)。历史实验文档和固定运行副本未删除。

## 当前工程实际状态

`demo_project/upright_rl_demo`当前配置configs/round19.json。仅安全真实两轮阶段补充大误差成本.25×(1−exp(−(原滤波误差/.3)²))奖励/秒，原精细成本/正奖励保持，总速度成本上界1.75<支撑2/秒。109项CPU、固定第18轮actor物理/控制/退出/评分逐项一致且奖励差符合独立公式、128环境4→8恢复、32例短训重载和默认2048环境2更新通过。旧actor只作开发工程检查，不进入PPO；实际全部能力尚未通过。

入口见[demo README](../demo_project/upright_rl_demo/README.md)、[需求基线](current_scope.md)、[重建计划](../demo_project/upright_rl_demo/docs/rebuild_plan.md)、[架构边界](../demo_project/upright_rl_demo/docs/architecture.md)。重整验收见[upright_reorganization_20261030.md](upright_reorganization_20261030.md)。

## 后续工作顺序

**新编号：** 系列upright_restart_20261030，当前upright_restart_round19；控制`demo_project/runs/_upright_restart_round19`，运行`upright_restart_round19_s101`/`upright_restart_round19_s102`。实际PPO、随机初始化、空历史模型列表及独立run_id已核验；运行中无eval_finished正常。第18轮独立/开发与请求/执行/实测轨迹已审查，旧事件已handled，不重复启动。

用户最新明确：需求已经清晰，历史实现的筛选和设计由agent判断；优先以较短、多批次demo训练快速验证实现。1000更新不是最终交付上限，正式从零流程可按收敛和验收需要使用10000或更多更新。本次为预算与职责澄清，未重新启动已取消队列。

本批完成后先读launch_status/eval_status/eval_finished、实际进程、缺失/失败与显存中断，区分开发课程检查和独立验证。起立→静止循环→60秒低速；已训练更高课程且低速物理完整至少26/32时再做该阶段60秒验证，其余显式跳过。低速基本不足仍记录，课程晋级的连续两次80%基本门槛不变。读取请求/执行/实测与成功/失败，尤其过渡动作质量；按结果修改并过工程门槛后再从零迭代。完成/异常事件已接回当前会话，退出观察器为一次性PID等待。

所有用户目标保持：普通四轮全能力独立、Space直线起立且少调整、两轮共同髋自主平衡、落地只增默认髋位且继承速度和交接、意外退出不自动重起、±25rad/s目标及源模型只读、当前demo单份1000更新、最终交付预算按收敛配置且可达10000或更多、ONNX后做真实MuJoCo闭环。正式训练目录保持空，用户亲自体验认可后才讨论迁移。

## 历史资料的使用边界

历史第十几批完整流程与C58桌面体验作为事实保留，早期有效源码可核查后复用。实际谱系显示完整循环第1批已有actor迁移、第12批已有冻结父网络、第17批继承第16批；不能简单以20批作为依赖分界。静止v5有独立从零训练证据，可用于候选来源审查；选定基础前仍需核查代码与模型适用性。第24/25批失败、v28工程结果均只留档；第26批为未完成的取消实验。旧移动第五轮及以后废弃结果仍不恢复。历史编号见[谱系](upright_training_lineage_20261023.md)，不需要全读docs或runs。
