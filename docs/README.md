# 项目文档索引

先读[AGENTS](../AGENTS.md)与[当前交接](current_handoff.md)，再按问题打开对应资料。不要全读docs或runs；根README只用于部署使用。

| 问题 | 入口 |
|---|---|
| 当前任务、运行状态、接手顺序 | [当前交接](current_handoff.md) |
| 全部用户目标和约束 | [需求基线](current_scope.md) |
| upright当前21r2设计、门槛和执行 | [21r2执行](upright_restart_round21r2_execution.md)、[21r1终局与分析](upright_restart_round21r1_results.md) |
| 第20轮停车问题与对照依据 | [归因复检](upright_round20_rollback_landing.md)、[交接修正](upright_round20_handoff_v1.md) |
| 当前MuJoCo体验与未解决问题 | [第20轮交接版及桌面反馈](upright_round20_handoff_sim2sim.md)、[使用说明](../rl_project/std_s2s_upright_project/README.md) |
| 重启系列历程、废弃记录 | [新系列1—20及后续索引](upright_restart_history.md)；按需读取 |
| 重启前旧upright记录在哪里 | [服务器history归档说明](upright_history.md)；不进入日常接手上下文 |
| repo、容器、独立分支与双端协作 | [工程管理](engineering_handoff.md)、[demo索引](../demo_project/README.md)、[正式项目索引](../rl_project/README.md) |
| 固定姿态恢复及参考来源 | [MuJoCo固定恢复](fixed_pose_recovery.md)、[参考项目选择](reference_selection.md)、[已关闭参考recovery](reference_recovery_validation.md) |
| 跳跃预研与早期调查 | [跳跃预研](jump_prestudy.md)、[历史资料](reports/) |

- Step从第14轮继续修正单次抬轮与停车：Restart16两路辅助均0/64，Restart17改持续停车目标，经物理门槛后两种子512×200从零验证。见[当前终局与迭代](step_restart17_20261103.md)；辅助不迁移，MuJoCo仍保留第20轮。

- Step历史回退：[暂停/回退证据](step_rollback14_20261103.md)，[第20轮终局](step_round20_results.md)、[第21轮取消记录](step_round21_execution.md)保留追溯；旧15–21均已废弃，不再作为优化基线。

- Step桌面预览：[暂停与第14轮回退](step_rollback14_20261103.md)。用户澄清训练回退不包含sim2sim，默认保留第20轮seed71供观察；桌面仍碰沿、上台后再抬头且不停，完整目标未通过。

## 写入规则

- 长期目标和约束写current_scope；当前状态只写current_handoff；固定方案、失败和结果写对应轮次文档及状态文件。
- AGENTS只保留影响当前决策的里程碑，README用于导览与使用，不逐轮追加进度。
- 完成/退出0不等于通过；区分用户确认、实测、开发干预和推测。过时状态移出当前入口，原始失败证据保留。
- 新系列历史按需查阅，旧系列只在服务器history中检索。归档不重写Git历史，也不恢复旧运行授权。
- 使用相对链接，不写主机地址、账户、密钥或代理配置。
