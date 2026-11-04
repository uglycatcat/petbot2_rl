# 项目文档索引

先读[AGENTS](../AGENTS.md)与[当前交接](current_handoff.md)，再按问题打开对应资料。不要全读docs或runs；根README只用于部署使用。

| 问题 | 入口 |
|---|---|
| 当前任务、运行状态、接手顺序 | [当前交接](current_handoff.md) |
| 全部用户目标和约束 | [需求基线](current_scope.md) |
| upright最新结果与训练暂停 | [21r6终局](upright_restart_round21r6_results.md)、[本轮设计与门槛](upright_restart_round21r6_execution.md)；用户要求先不开下一轮 |
| 第20轮停车问题与对照依据 | [归因复检](upright_round20_rollback_landing.md)、[交接修正](upright_round20_handoff_v1.md) |
| 当前MuJoCo体验与未解决问题 | [21r5用户请求预览](upright_restart21r5_sim2sim.md)、[使用说明](../rl_project/std_s2s_upright_project/README.md) |
| 重启系列历程、废弃记录 | [新系列1—20及后续索引](upright_restart_history.md)；按需读取 |
| 重启前旧upright记录在哪里 | [服务器history归档说明](upright_history.md)；不进入日常接手上下文 |
| repo、容器、独立分支与双端协作 | [工程管理](engineering_handoff.md)、[demo索引](../demo_project/README.md)、[正式项目索引](../rl_project/README.md) |
| 固定姿态恢复及参考来源 | [MuJoCo固定恢复](fixed_pose_recovery.md)、[参考项目选择](reference_selection.md)、[已关闭参考recovery](reference_recovery_validation.md) |
| 跳跃预研与早期调查 | [跳跃预研](jump_prestudy.md)、[历史资料](reports/) |

- Step当前：[v2第1批完整训练](step_v2_round01_20261104.md)。用户在[重构工程通过](step_refactor_20261104.md)后重新授权，GPU0/1两种子各512×500从零学习完整任务；最终能力待独立评估。

- Step历史回退：[暂停/回退证据](step_rollback14_20261103.md)，[第20轮终局](step_round20_results.md)、[第21轮取消记录](step_round21_execution.md)保留追溯；旧15–21均已废弃，不再作为优化基线。

- Step桌面预览：[暂停与第14轮回退](step_rollback14_20261103.md)。用户澄清训练回退不包含sim2sim，默认保留第20轮seed71供观察；桌面仍碰沿、上台后再抬头且不停，完整目标未通过。

## 写入规则

- 长期目标和约束写current_scope；当前状态只写current_handoff；固定方案、失败和结果写对应轮次文档及状态文件。
- AGENTS只保留影响当前决策的里程碑，README用于导览与使用，不逐轮追加进度。
- 完成/退出0不等于通过；区分用户确认、实测、开发干预和推测。过时状态移出当前入口，原始失败证据保留。
- 新系列历史按需查阅，旧系列只在服务器history中检索。归档不重写Git历史，也不恢复旧运行授权。
- 使用相对链接，不写主机地址、账户、密钥或代理配置。
