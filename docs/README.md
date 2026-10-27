# 项目文档索引

先读根目录 [AGENTS.md](../AGENTS.md) 的硬规则和里程碑，再读[当前交接](current_handoff.md)。只按手头问题打开下面相应文档；历史实验与 `runs/` 不需要全读。根目录 README 用于部署和使用，不承载进度。

| 问题 | 文档 |
|---|---|
| MuJoCo 键盘体验、正式项目与成熟度验证 | [sim2sim 方案](upright_sim2sim_20261027.md)、[正式项目索引](../rl_project/README.md) |
| 当前正在做什么、状态文件在哪里、下一步怎么接 | [当前交接](current_handoff.md) |
| 三个功能目标、控制接口与用户已确认的约束 | [需求基线](current_scope.md) |
| repo、容器与版本管理 | [工程说明](engineering_handoff.md)的管理章节；[demo 索引](../demo_project/README.md)。工程说明其余训练接口主要为历史 recovery 原型。 |
| 静止、旧移动、修复重建、完整循环的父子关系 | [训练谱系](upright_training_lineage_20261023.md) |
| 第六版同配置续长的方案与终局 | [长训方案](upright_cycle_v6_long.md)、[长训终局](upright_cycle_batch6_long_results.md)及[原第六批终局](upright_cycle_batch6_results.md) |
| 第十五批终局与当前执行边界 | [第十五批结果](upright_cycle_batch15_results.md)、[当前交接](current_handoff.md) |
| 瞬时联合0%的含义、平衡波动与遥控验收 | [验收口径复核](upright_tracking_acceptance_review_20261026.md)，分析建议，未变更正式标准 |
| 放宽精度、两轮不跌倒优先的最新需求与改法 | [稳定优先方案](upright_stability_first_plan_20261026.md)，已获实施与训练授权，状态见交接 |
| 过渡偏航/落地修正与全模型范围对齐 | [修正与验收](upright_cycle_correction_20261027.md)，当前暂停部署 |
| 稳定优先版本与新功能验收 | [第十六版](upright_cycle_v16.md)；部署状态见[交接](current_handoff.md) |
| 最新起立修正与独立范围结果 | [第十七批结果](upright_cycle_batch17_results.md)、[固定策略验证](upright_cycle17_validation.md)、[交接顺序修正](upright_handoff_alignment_v18.md)、[独立范围/起立诊断](upright_handoff18_validation.md)、[起立专用修正v19](upright_rise_adapter_v19.md)、[第十八批起立训练](upright_cycle_v19_training.md)、[第十八批终局](upright_cycle_batch18_results.md)、[起立幅度诊断](upright_rise_authority_diagnostic.md)、[同配置续训](upright_cycle_v19_long.md)、[续训终局](upright_cycle_batch18_long_results.md)；原对照见[第十七版](upright_cycle_v17.md) |
| 稳定两轮局部反馈设计与验收 | [第十五版](upright_cycle_v15.md) |
| 第十四批完整结果与后续诊断 | [第十四批终局](upright_cycle_batch14_results.md) |
| 转向内环周期/增益对照 | [第十四版](upright_cycle_v14.md)，实际状态见[交接](current_handoff.md) |
| 瞬时容差与转向修正对照 | [第十三版](upright_cycle_v13.md)，实际状态见[交接](current_handoff.md) |
| 第十三批完整结果与后续诊断 | [第十三批终局](upright_cycle_batch13_results.md) |
| 第十二批完整结果 | [第十二批终局](upright_cycle_batch12_results.md) |
| 第十二版稳定两轮精度修正对照 | [第十二版](upright_cycle_v12.md)，实际状态见[交接](current_handoff.md) |
| 第十一批归一化对照终局 | [第十一批结果](upright_cycle_batch11_results.md)，当前诊断见[交接](current_handoff.md) |
| actor统计与等价迁移设计 | [第十一版](upright_cycle_v11.md)，实际状态见[交接](current_handoff.md) |
| 第十批终局和前向控制诊断 | [第十批结果](upright_cycle_batch10_results.md)，实际进程见[交接](current_handoff.md) |
| 第十版快速转向反馈对照与工程验收 | [第十版](upright_cycle_v10.md)；已终局，固定设计与结果分开阅读 |
| 第九版地面航向与瞬时跟踪对照 | [第九版](upright_cycle_v9.md)；新旧坐标分开报告，实时状态见[当前交接](current_handoff.md) |
| 第九批修复采样与航向后的实际效果 | [第九批终局](upright_cycle_batch9_results.md)；先做固定策略诊断，再决定下一批 |
| 第八批效果和子步诊断依据 | [第八批终局](upright_cycle_batch8_results.md) |
| 第八版转向反馈与精细探索 | [第八版](upright_cycle_v8.md)、[第七批终局](upright_cycle_batch7_results.md)；实时状态见[当前交接](current_handoff.md) |
| 第七版持续遥控设计与验收 | [第七版](upright_cycle_v7.md)、[部署计划](experiments/upright_cycle_batch7.json)；运行状态以[当前交接](current_handoff.md)为准 |
| 差速目标为何没有形成预期转向、下一步诊断 | [受控差速响应](upright_yaw_diagnostic_20261023.md)；含批次事件续接说明。 |
| 早期静止起立成果 | [静止第五版](upright_v5_training.md)；任务设计见[upright.md](upright.md) |
| 旧移动结果为何退出优化基线 | [夜间审查](upright_overnight_audit_20261021.md)、[修复记录](upright_repair_20261021.md) |
| 完整循环如何演变 | [完整循环设计](upright_cycle_v1.md)；具体某批从对应 `upright_cycle_v*.md` 与 `upright_cycle_batch*_results.md` 查 |
| 固定姿态恢复与参考项目 | [MuJoCo 固定恢复](fixed_pose_recovery.md)、[已关闭参考 recovery](reference_recovery_validation.md)、[参考项目选择](reference_selection.md) |
| 跳跃预研、早期调查 | [跳跃预研](jump_prestudy.md)、[历史资料](reports/) |

## 写入规则

- 需求与长期约束写 `current_scope.md`；运行中的固定方案、评估与失败分析写对应实验文档和状态文件；当前接手顺序写 `current_handoff.md`。
- 只有改变后续决策的重要节点才进入 AGENTS 的里程碑表：确认的需求变化、独立评估后的终局、旧结果失效、重大修复或停止决策。轮询、ETA与中间检查点留在实验记录，不重复抄到索引。
- 修改 demo 入口和工具职责时更新对应 demo README；它用于项目导览，不写逐批进度。区分用户确认、实测与推测，过时结论注明历史边界。
- 早期实验文档保留当时的运行状态与判断；其中的“当前”“定时启用”“下一步”均按文档日期理解，不覆盖[当前交接](current_handoff.md)。
- 文档使用相对链接；仓库公开，不写主机地址、账户、密钥或代理配置。
