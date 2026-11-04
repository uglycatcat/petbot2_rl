# upright重启系列历程（按需查阅）

系列起点：2026-10-30，`upright_restart_20261030`。当前状态见[交接](current_handoff.md)，本页不作为全部必读清单。旧完整循环batch与本系列round编号不同，不能混用。

| 轮次 | 设计/执行 | 结果与用途 |
|---|---|---|
| 1 | [执行](upright_restart_round01_execution.md) | [结果](upright_restart_round01_results.md)，历史从零实验，按需追溯 |
| 2 | [执行](upright_restart_round02_execution.md) | [结果](upright_restart_round02_results.md)，历史从零实验，按需追溯 |
| 3 | [执行](upright_restart_round03_execution.md) | [结果](upright_restart_round03_results.md)，历史从零实验，按需追溯 |
| 4 | [执行](upright_restart_round04_execution.md) | [结果](upright_restart_round04_results.md)，历史从零实验，按需追溯 |
| 5 | [执行](upright_restart_round05_execution.md) | [结果](upright_restart_round05_results.md)，历史从零实验，按需追溯 |
| 6 | [执行](upright_restart_round06_execution.md) | [结果](upright_restart_round06_results.md)，历史从零实验，按需追溯 |
| 7 | [执行](upright_restart_round07_execution.md) | [结果](upright_restart_round07_results.md)，历史从零实验，按需追溯 |
| 8 | [执行](upright_restart_round08_execution.md) | [结果](upright_restart_round08_results.md)，历史从零实验，按需追溯 |
| 9 | [执行](upright_restart_round09_execution.md) | [结果](upright_restart_round09_results.md)，历史从零实验，按需追溯 |
| 10 | [执行](upright_restart_round10_execution.md) | [结果](upright_restart_round10_results.md)，历史从零实验，按需追溯 |
| 11 | [执行](upright_restart_round11_execution.md) | [结果](upright_restart_round11_results.md)，历史从零实验，按需追溯 |
| 12 | [执行](upright_restart_round12_execution.md) | [结果](upright_restart_round12_results.md)，历史从零实验，按需追溯 |
| 13 | [执行](upright_restart_round13_execution.md) | [结果](upright_restart_round13_results.md)，历史从零实验，按需追溯 |
| 14 | [执行](upright_restart_round14_execution.md) | [结果](upright_restart_round14_results.md)，历史从零实验，按需追溯 |
| 15 | [执行](upright_restart_round15_execution.md) | [结果](upright_restart_round15_results.md)，历史从零实验，按需追溯 |
| 16 | [执行](upright_restart_round16_execution.md) | [结果](upright_restart_round16_results.md)，历史从零实验，按需追溯 |
| 17 | [执行](upright_restart_round17_execution.md) | [结果](upright_restart_round17_results.md)，历史从零实验，按需追溯 |
| 18 | [执行](upright_restart_round18_execution.md) | [结果](upright_restart_round18_results.md)，历史从零实验，按需追溯 |
| 19 | [执行](upright_restart_round19_execution.md) | [结果](upright_restart_round19_results.md)，历史从零实验，按需追溯 |
| 20 | [执行](upright_restart_round20_execution.md) | [结果](upright_restart_round20_results.md)，当前认可的运动基准，停车仍未过 |
| 21 | [执行](upright_restart_round21_execution.md) | [结果](upright_restart_round21_results.md)，原始记录；已退出当前优化路线 |
| 22 | [执行](upright_restart_round22_execution.md) | [结果](upright_restart_round22_results.md)，原始记录；已退出当前优化路线 |
| 23 | [执行](upright_restart_round23_execution.md) | [结果](upright_restart_round23_results.md)，原始记录；已退出当前优化路线 |
| 24 | [执行](upright_restart_round24_execution.md) | [结果](upright_restart_round24_results.md)，原始记录；已退出当前优化路线 |
| 25 | [执行](upright_restart_round25_execution.md) | [结果](upright_restart_round25_results.md)，原始记录；已退出当前优化路线 |
| 26 | [执行](upright_restart_round26_execution.md) | [结果](upright_restart_round26_results.md)，原始记录；已退出当前优化路线 |
| 27 | [执行](upright_restart_round27_execution.md) | 用户取消，不恢复，无1000更新终局 |
| 21r1 | [执行](upright_restart_round21r1_execution.md) | [终局](upright_restart_round21r1_results.md)：落地局部改善但高范围退步，拒绝升级 |
| 21r2 | [执行](upright_restart_round21r2_execution.md) | [终局](upright_restart_round21r2_results.md)：无成本高范围29/32，成本组0/32，停止成本课程路线 |
| 21r3 | [执行](upright_restart_round21r3_execution.md) | [终局](upright_restart_round21r3_results.md)：102高范围31/32、落地改善，MuJoCo残余与速度目标仍未过 |
| 21r4 | [执行](upright_restart_round21r4_execution.md) | [终局](upright_restart_round21r4_results.md)：静止后退约1mm，低速/零令响应未保住，桌面20保持 |
| 21r5 | [执行](upright_restart_round21r5_execution.md) | [终局](upright_restart_round21r5_results.md)：低速基本31/26，落地仍动、高速无一致收益 |
| 21r6 | [执行](upright_restart_round21r6_execution.md) | [终局](upright_restart_round21r6_results.md)：落地总移动改善，运动保留未过；用户暂停后续训练 |

[编号与旧21—26废弃/27取消](upright_restart_discarded_rounds_20261103.md)。这些记录和固定运行副本保留，不能重启旧队列或作为新优化来源。

补充入口：[第1轮计划](upright_restart_round01_plan.md)、[工程重建来源](upright_reorganization_20261030.md)、[第20轮桌面反馈](upright_restart20_desktop_feedback_20261102.md)、[落地控制权澄清](upright_restart_landing_motion_20261102.md)、[当前交接版发布](upright_round20_handoff_sim2sim.md)。原第26轮发布仅作[废弃路线历史](upright_restart26_sim2sim.md)。

当前`demo_project/runs/`保留新系列及其工程检查、当前诊断、原第20轮基准和在用依赖。重启前旧系列见[服务器归档说明](upright_history.md)，不混入当前历程。
