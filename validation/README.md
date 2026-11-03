# MuJoCo验证索引

当前默认是新系列第20轮seed102原ONNX＋正常交接修正。用户认可流程与运动体验，并确认交接修改局部改善；**零输入落地仍有后退，完整能力未通过。** 导出、工程、独立评估和桌面体验须分别判断。

| 记录 | 证据与边界 |
|---|---|
| [当前交接修正版](20261103_round20_handoff/review.json) | 三条42秒场景、普通持令/松键/退出、180组源码对齐与窗口检查；后退仍2.6–3.6cm |
| [第20轮首次发布](20261102_restart20/review.json) | 单ONNX、180组观测/目标、27姿态几何及两个短流程；不是双向0.75或宽初态通过 |
| [首次零输入锁存修复](20261102_idle_fix/review.json) | 仅解决最终不停，不能证明落地后立即静止 |
| [交接输入提前响应](20261102_prompt_handoff/review.json) | 历史中间修正，后退仍12–14cm，已由当前交接版替代 |
| [废弃第26轮预览](20261103_restart26/review.json) | 原运行证据和失败保留，不作当前优化或推荐基线 |

原始轨迹位于本项目ignored的outputs/artifacts中；JSON摘要、源commit和SHA保留原样。旧系列20261027—20261029的验证JSON作为现存历史策略包的审计附件保留，详细原文在服务器`history/upright/archival_20261103/before/rl_project/std_s2s_upright_project/validation/README.md`，不列为默认接手材料。

主仓库[当前发布与桌面反馈](../../../docs/upright_round20_handoff_sim2sim.md)、[新系列历程](../../../docs/upright_restart_history.md)、[旧系列归档说明](../../../docs/upright_history.md)。不要把任何历史版本的“默认”“已通过”理解为当前状态。
