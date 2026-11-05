# 项目文档索引

先读[AGENTS](../AGENTS.md)与[当前交接](current_handoff.md)，再按问题打开对应资料。不要全读docs或runs；根README只用于部署使用。

| 问题 | 入口 |
|---|---|
| 当前任务、运行状态、接手顺序 | [当前交接](current_handoff.md) |
| 项目需求 | [upright需求](current_scope.md)、[STEP需求](step_scope.md)；按项目阅读 |
| STEP当前状态、结果与待讨论问题 | [STEP交接](step_current.md)；v2已暂停 |
| STEP当前实现与管理 | [实现说明](../demo_project/step_rl_demo/docs/architecture.md)、[管理流程](../demo_project/step_rl_demo/docs/automation.md) |
| STEP旧轮次与旧说明在哪里 | [STEP history定位](step_history.md)；仅服务器本地、按需查指定文件 |
| upright当前版本、效果与未解决问题 | [1104当前说明](upright_1104.md)；4096×5000单次训练完成，后续训练停用 |
| upright完整可复现训练代码 | [临时1104工程](../rl_project/train_upright_project_1104/README.md)；std_train仍预留为空 |
| upright现有MuJoCo体验 | [sim2sim使用](../rl_project/std_s2s_upright_project/README.md)；默认`upright1104_5000_s102`，原21r5保留对照 |
| upright本次5000更新日志和检查点 | [1104运行记录](../rl_project/train_upright_project_1104/README.md#本次运行与日志)；项目内logs/outputs，仅服务器保存 |
| upright逐轮文档、runs、archive分支在哪里 | [服务器history定位](upright_history.md)；全部仅本地保存、按需检索 |
| repo、容器、独立分支与双端协作 | [工程管理](engineering_handoff.md)、[demo索引](../demo_project/README.md)、[正式项目索引](../rl_project/README.md) |
| 固定姿态恢复及参考来源 | [MuJoCo固定恢复](fixed_pose_recovery.md)、[参考项目选择](reference_selection.md)、[已关闭参考recovery](reference_recovery_validation.md) |
| 跳跃预研与早期调查 | [跳跃预研](jump_prestudy.md)、[历史资料](reports/) |

## 写入规则

- 按项目区分用户需求、实现选择、实测结果和待验证假设。STEP需求写step_scope、当前状态写step_current，实现写其architecture；根交接只链接。详细过程保留在运行证据或history，不逐轮累加到默认入口。
- AGENTS只保留影响当前决策的里程碑，README用于导览与使用，不逐轮追加进度。
- 完成/退出0不等于通过；区分用户确认、实测、开发干预和推测。过时状态移出当前入口，原始失败证据保留。
- upright历史与重构前STEP资料只在服务器history中按需检索；旧假设和授权不自动继承。归档不重写Git历史，也不恢复旧运行授权。
- 使用相对链接，不写主机地址、账户、密钥或代理配置。
