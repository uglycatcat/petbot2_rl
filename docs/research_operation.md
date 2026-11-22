# 三路自动探索运行入口

当前正式配置：`configs/research_20261122/plan.json`。所有训练从零，旧5cm/7cm策略只作对照。首先阅读[研究需求](research_scope.md)和[当前交接](current.md)。

## 状态与证据

- `control/research_control.json`：新研究开关，和旧`control/control.json`相互独立。
- `control/research_control_process.json`：宿主编排PID、当前运行目录。
- `control/research_status.json`：各lane当前阶段、实际训练更新、显存和最近结果。
- 当前运行目录`series.json`：已发布冻结提交、每路配置和资产哈希。
- 每路`trialNNN/config.json`及`tensorboard/`：本次从零配置与曲线；`resources.jsonl`是实际GPU占用。
- 每路`eval_u*/`：独立64回合、原始物理轨迹、代表动作和行为指标。训练后先读取曲线，再独立评估第201、第501和最后更新（存在时）。
- 每路`review.json`、`video/execution.mp4`：所选策略、质量相对比值、轨迹差异与实际视频。满足候选门槛才导出，导出后另起进程复核。
- `catalog.json`：各trial选择结果；`frontier.json`：已观察到成功的最高高度与独立导出评估≥90%的最高高度。两者都不是物理极限。

仅物理支撑与轨迹证据决定任务成功。`different_behavior_candidate`只是轨迹描述距离超过工程阈值，必须结合视频/接触顺序复核才能称作不同动作。质量比值比较头角、远处车架角与目标变化率，不替代成功率和逐条轨迹审查。

## 调度与停止

新批次由已发布干净源码启动：

```bash
python3 -u tools/research_campaign.py --plan configs/research_20261122/plan.json
```

当前已有活跃编排时不要重复运行。控制文件有独占锁；启动后从`git archive`冻结源码及源模型派生资产，所有容器使用该固定副本。禁止修改运行中的`source/`、`assets/`或trial配置。工程短流程使用独立`research_engineering_control.json`，不代表正式训练。

设置`enabled=false`或`automatic_iteration=false`会停止后续派发；需要当前训练也在检查点边界退出时同时设置`stop_current=true`。对控制文件应写临时文件后原子替换，避免读取半份JSON。只能调度本项目专用容器，不能清理其他任务和其他用户进程。

每卡共享准入预留至少1536MiB，并计入同卡尚未达到峰值的任务。显存压力只选择本卡最新进入的一路先退出；仅同一trial的真正资源中断可从保存点恢复，最多三次恢复。工程错误或超时隔离该lane并记录错误，其他lane继续；不要将隔离结果当作学习失败。已完成trial不恢复，下一trial总是新seed随机初始化。

首个自动探索窗口每lane最多8个trial，合计最多48个。质量路线根据独立成功率和质量指标调节软约束；多样性路线换种子、周期和中间奖励；高度路线根据独立成功率递进/复试/细化。窗口结束关闭新控制并保留结果，不声明研究目标或高度极限已经完成。后续设计依据实际结果更新，并重新冻结新源码。
