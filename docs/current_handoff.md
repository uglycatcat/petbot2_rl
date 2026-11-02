# 当前交接：upright第20轮已验收，第21轮60ms长指令保持已从零启动

更新：2026-11-02。先读AGENTS.md，再读本页。upright按最新GPU2/3从零训练授权执行，旧第26批不恢复；完成事件先核查实际状态再分析迭代。

## 最新指令与执行边界

**2026-11-02新增用户授权已执行：** sim2sim默认更新为新第20轮seed102的1000更新单ONNX包，旧C58权重保留供显式对照。MuJoCo两个短流程和接口/普通控制/退出/窗口检查有效，实际轮速超调仍在；不能替代全范围和用户验收。第21轮训练不变。同步版本与启动见[新包发布](upright_restart20_sim2sim.md)。

用户持续授权单Actor–Critic从零训练、评估和事件迭代，仅GPU2/3。第20轮两份1000更新/7独立评估有效，另1按计划跳过，无缺失/OOM/恢复；seed101卡在静止落地课程，静止/低速0，seed102低速32/32、高范围29/32，前.487/后.535m/s、实际轮峰128.801rad/s。30ms没有保持跨种子流程，不保留。第21轮固定3c1bd5e恢复第18轮60ms/32步/GAE .95，仅对第3级起长训练回合增加2秒指令保持，最高4→6秒；短回合/低课程/原评估保持。119项CPU、固定原评估逐帧一致、实际训练采样分支及短训恢复/重载/默认规模有效，GPU2/3各2048×1000已从零进入PPO。见[第20轮终局](upright_restart_round20_results.md)、[第21轮执行](upright_restart_round21_execution.md)。无历史训练权重、周期定时器/goal。

用户最新纠正：需求与期望效果已明确，早期已实现基本起立、两轮与落地，应核查并复用有效代码和经验，重点是训练代码迭代效率，不必全部重新设计。此前将整个旧实现永久排除的解释范围过大，已纠正；完整封存仍保留；已选择性复用静止v5资产/分布代码并实现独立完整循环。新训练仍禁止历史权重和冻结父依赖。

**此前仅修改的暂停已被最新自动迭代授权替代。** 不恢复第26批、不加载历史模型；既有C58桌面包未改动，仅作体验基准，不作新工程能力证明。

## 已完成的停止与封存

- 第26批两路训练及启动器、完成/失败观察器均已退出。训练退出143是用户取消，无显存/PhysX/Python中断；请求停止时日志约454/457更新，各保存至model_400.pt，未完成1000更新，未做终局评估。
- 控制目录`demo_project/runs/_upright_cycle_batch26`已将launch_status/eval_status/eval_finished结算为cancelled_by_user，并保存取消前状态、cancellation和completion_event_reconciliation。不得当失败自动恢复，也不得当完成或能力通过。
- 旧demo源码b357dd7已推送归档分支`archive/upright_rl_demo_pre_rebuild_20261030`，保留全部Git历史。完整本地目录保存在`demo_project/runs/_upright_reorganization_20261030/checkout_before_rebuild`；180个跟踪文件哈希一致，生成物及未跟踪文件也保留。
- 以前的需求/交接分别原样保存为[current_scope_archive_20261030.md](current_scope_archive_20261030.md)、[current_handoff_archive_20261030.md](current_handoff_archive_20261030.md)。历史实验文档和固定运行副本未删除。

## 当前工程实际状态

`demo_project/upright_rl_demo`当前配置configs/round21.json。恢复60ms/32步/GAE .95和原五帧时间跨度，精细奖励/Beta512/actor195/critic219保持。long_command_hold_extra_s=2只影响第3级起长训练回合，最高4→6秒；短回合、低课程和全部评估原协议保持，实际遥控接口仍按原周期接收新指令。critic指令年龄按实际保持时间归一化，日志/轨迹增加command_dwell_s。119项CPU、固定第18轮原评估物理/奖励/评分逐帧一致、实际训练采样分支、128环境4→8恢复/重载及默认2048×2有效。工程不等于新策略能力；轨迹按实际时间戳分析，本轮恢复60ms。

入口见[demo README](../demo_project/upright_rl_demo/README.md)、[需求基线](current_scope.md)、[重建计划](../demo_project/upright_rl_demo/docs/rebuild_plan.md)、[架构边界](../demo_project/upright_rl_demo/docs/architecture.md)。重整验收见[upright_reorganization_20261030.md](upright_reorganization_20261030.md)。

## 后续工作顺序

**新编号：** 系列upright_restart_20261030，当前upright_restart_round21；控制`demo_project/runs/_upright_restart_round21`，运行`upright_restart_round21_s101`/`upright_restart_round21_s102`。实际PPO、随机初始化、空历史模型列表、独立run_id及工程指纹均已核验；无eval_finished属于运行未完成。第20轮已handled，旧队列与全部工程进程已退出，不重复启动。

用户最新明确：需求已经清晰，历史实现的筛选和设计由agent判断；优先以较短、多批次demo训练快速验证实现。1000更新不是最终交付上限，正式从零流程可按收敛和验收需要使用10000或更多更新。本次为预算与职责澄清，未重新启动已取消队列。

本批完成后先读launch_status/eval_status/eval_finished、实际进程、缺失/失败与显存中断，区分开发课程检查和独立验证。起立→静止循环→60秒低速；已训练更高课程且低速物理完整至少26/32时再做该阶段60秒验证，其余显式跳过。低速基本不足仍记录，课程晋级的连续两次80%基本门槛不变。读取请求/执行/实测与成功/失败，尤其过渡动作质量；按结果修改并过工程门槛后再从零迭代。完成/异常事件已接回当前会话，退出观察器为一次性PID等待。

所有用户目标保持：普通四轮全能力独立、Space直线起立且少调整、两轮共同髋自主平衡、落地只增默认髋位且继承速度和交接、意外退出不自动重起、±25rad/s目标及源模型只读、当前demo单份1000更新、最终交付预算按收敛配置且可达10000或更多、ONNX后做真实MuJoCo闭环。正式训练目录保持空，用户亲自体验认可后才讨论迁移。

## 历史资料的使用边界

历史第十几批完整流程与C58桌面体验作为事实保留，早期有效源码可核查后复用。实际谱系显示完整循环第1批已有actor迁移、第12批已有冻结父网络、第17批继承第16批；不能简单以20批作为依赖分界。静止v5有独立从零训练证据，可用于候选来源审查；选定基础前仍需核查代码与模型适用性。第24/25批失败、v28工程结果均只留档；第26批为未完成的取消实验。旧移动第五轮及以后废弃结果仍不恢复。历史编号见[谱系](upright_training_lineage_20261023.md)，不需要全读docs或runs。
