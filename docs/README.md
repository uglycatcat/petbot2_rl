# 项目文档索引

先读[AGENTS](../AGENTS.md)与[当前交接](current_handoff.md)，再按问题打开对应资料。不要全读docs或runs；根README只用于部署使用。

| 问题 | 入口 |
|---|---|
| 当前任务、运行状态、接手顺序 | [当前交接](current_handoff.md) |
| 全部用户目标和约束 | [需求基线](current_scope.md) |
| upright当前版本、效果与未解决问题 | [1104冻结说明](upright_1104.md)；21r5 seed102，全部upright训练停止 |
| upright完整可复现训练代码 | [临时1104工程](../rl_project/train_upright_project_1104/README.md)；std_train仍预留为空 |
| upright现有MuJoCo体验 | [sim2sim使用](../rl_project/std_s2s_upright_project/README.md)；保持21r5，未因归档修改 |
| upright逐轮文档、runs、archive分支在哪里 | [服务器history定位](upright_history.md)；全部仅本地保存、按需检索 |
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
- upright全部历史只在服务器history中按需检索。归档不重写Git历史，也不恢复旧运行授权。
- 使用相对链接，不写主机地址、账户、密钥或代理配置。
