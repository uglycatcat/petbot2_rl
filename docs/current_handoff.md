# 当前交接：PetBot 完整运动循环起立

更新：2026-10-23。本页只放当前决策、证据入口与接手顺序；运行状态以磁盘状态和实际进程为准。

## 目标与当前决定

三个策略按起立、低台阶、跌倒恢复推进；目前仅起立是活跃训练线。目标是四轮正常行驶时接管，允许为平衡减速但不预先停车，前轮抬起后以后轮支撑继续响应前后与转向遥控；切回普通控制时由同一策略使前轮落地并完成平顺交接。车架输出位置、左右后轮输出速度。详见[需求基线](current_scope.md)。静止起立第五版已有默认初态的阶段成果，但不能当作运动循环达标。

完整循环第六批固定任务提交 `6b5412b`：独立种子交接 A/B 分别 381/380（各384），按用户放宽落地速度/力门槛后完整合格仍各 1/384，主要缺口是两轮阶段遥控跟踪与平衡。[终局结果](upright_cycle_batch6_results.md)有原始/新版口径及分项。600 iteration 是预算，不是收敛证明；当前决定保持任务设计和奖励不变，从同组 `model_599` 完整恢复 actor、critic、normalizer、optimizer，续到 2399，检验训练预算。设计、固定配置和工程验收见[长训记录](upright_cycle_v6_long.md)。

## 当前固定批次与实时证据

| 对象 | 位置或说明 |
|---|---|
| 固定任务代码 | `6b5412b`；正式运行从 `demo_project/runs/` 中的固定副本执行，不从可维护 checkout 热改。 |
| 部署工具 | `b4769b3`；12组、四卡、检查点1000/1600/2399，原生及独立种子/宽范围评估计划共60项。 |
| 实验计划 | [`experiments/upright_cycle_batch6_long.json`](experiments/upright_cycle_batch6_long.json)。 |
| 部署状态 | `demo_project/runs/_upright_cycle_batch6_long/launch_status.json`。 |
| 评估状态 | 同目录的 `eval_status.json`、`eval_finished.json`（后者在队列全部结束前可能不存在）。 |
| 训练日志和检查点 | `launch_status.json` 的 `train` 项逐组记录运行名、GPU、PID和启动命令；据此定位对应 `runs/<run>/logs/` 与守护日志。 |
| 容量门槛 | 发新训练前检查其他用户任务、四卡计算利用率与整卡显存；目标计算利用率接近80%，整卡显存78%守护，不影响他人。 |

截至2026-10-23 14:49，`launch_status.json` 已记录12组进入PPO，12个守护PID仍存在；`eval_status.json` 登记1项，`eval_finished.json` 尚不存在。这只是带时间戳的快照；接手时必须重新核对进程、GPU、日志和失败/缺失项，不能把“队列启动”当作“评估通过”。**禁止重复启动当前批次。** 旧移动任务第五轮及之后的夜间结果已废弃，不用于优化，也不自动恢复第十轮。[批次谱系](upright_training_lineage_20261023.md)解释静止、旧移动、修复重建与完整循环四个编号序列。

用户已取消每日04/10/16/22定时续接：`demo_project/runs/_recurring_upright/config.json` 为 `enabled=false`，systemd 用户 timer 已停用。未经新的明确要求不得重新启用；既有长训与其评估队列继续。

## 接手顺序

1. 先读根目录 [AGENTS.md](../AGENTS.md) 的硬规则和里程碑，再读本页；不要遍历 `runs/` 或全部历史文档。
2. 检查本批 `launch_status.json`、`eval_status.json`、`eval_finished.json`、实际 PID、GPU 和日志。发现中断先诊断和恢复同批，不能盲目开新批。
3. 批次评估齐全后，按[第六批终局](upright_cycle_batch6_results.md)相同的独立口径比较1000/1600/2399。区分交接、两轮跟踪、平衡和新版完整合格；单看交接成功率或训练奖励不足以判定达标。
4. 如果长训仍无实质改善，再基于分项结果讨论奖励、任务与控制结构。先形成可核对的固定版本方案、GPU验收，再启动下一批。结论写对应实验文档；重大目标/有效性/终局变化才写 AGENTS 里程碑表。

## 文档树

| 需要解决的问题 | 只读这些入口 |
|---|---|
| 最新状态和下一动作 | 本页及该批状态 JSON、日志。 |
| 用户确认的需求和工程约束 | [需求基线](current_scope.md)、[工程约定](engineering_handoff.md)。 |
| 代码入口和产物位置 | [demo 导览](../demo_project/upright_rl_demo/README.md)、[demo 索引](../demo_project/README.md)。 |
| 本批设计与批次来源 | [长训记录](upright_cycle_v6_long.md)、[谱系](upright_training_lineage_20261023.md)。 |
| 历史失败及对照 | [文档索引](README.md)按问题进入对应实验文档；不全读。 |
