# upright工程重整：只继承需求，旧实现统一封存

2026-10-30用户先要求停止当前训练、重整工程，随后明确：继承式训练之后的代码不作为复用基础，后续澄清的需求仍有效。本次不是对旧奖励继续打补丁，也没有新开训练；旧实现统一退出当前开发入口。

## 停止与证据

第26批启动器、完成/失败观察器和两路训练均已停止，7个相关监督/宿主进程退出，容器内对应任务由唯一job token定向终止。训练退出143，显存守护未触发、PhysX/Python错误为空。请求停止时日志约seed60 454、seed61 457更新，各保留model_0/100/200/300/400；未保存的后续进度不冒称可恢复。该批未完成1000更新、未进入终局评估，状态是用户取消，不能结算为通过、完整失败终局或待自动恢复。

`runs/_upright_cycle_batch26`保留取消前状态与原始日志，当前launch_status/eval_status/eval_finished和completion_event_reconciliation已明确cancelled_by_user、禁止自动恢复。首次停止工具调用因Signal枚举转字符串不兼容而未发出信号，改用整数15后成功；该操作错误已记录在cancellation.json，没有启动重复任务。

## 封存完整性

旧demo源码固定`b357dd720ae5cc42e0c630476a0df007fe7dbc3e`，已推送同一仓库独立分支`archive/upright_rl_demo_pre_rebuild_20261030`，保留全部历史提交。本地完整目录迁至`demo_project/runs/_upright_reorganization_20261030/checkout_before_rebuild`，原180个跟踪文件逐项SHA256一致，未跟踪文件、派生资产与输出一起保留。`.git`及repo管理关系保持，主工作分支仍为`demoproj/upright_rl_demo`。

没有删除固定运行副本、检查点或轨迹，没有改动源模型及既有MuJoCo包。旧当前需求、交接及AGENTS入口分别原样保存到`current_scope_archive_20261030.md`、`current_handoff_archive_20261030.md`、`agents_context_archive_20261030.md`；其中包含整理前已有的本地文档修改。历史记录是追溯材料，不授予新训练权限，也不是新工程的实现或优化来源。

## 当前工程内容

当前开发目录只保留新写的项目入口、机器可读用户规格、重建顺序及架构边界。`project.py`只读取需求与源MJCF，核对50与25rad/s取小规则、左右髋限位及默认落地角，不启动仿真。`train.py`明确返回未实现且暂停，无旧代码回退路径；旧训练脚本、launch/watch工具、奖励和课程均不在当前开发目录。

用户最终目标未收缩：完整独立调用、普通四轮全能力、不同共同髋初态、直线起立少调整、双向连续0–0.75m/s、镜像响应、两轮自主共同髋、落地只增加默认髋位并继承速度/交接、异常退出不重起、ONNX后MuJoCo真实闭环、每份正式1000更新。见[纯需求基线](current_scope.md)。

新网络、动作分布、控制周期、奖励、课程、仿真训练实现和验收阈值尚未建立。旧静止、完整循环与C58体验的事实保持记录，但本次没有选择其中任何实现作为新版基础。禁止以“整理完成”宣称项目已能从零训练出结果。

## 验证与后续边界

本次仅做封存完整性、需求/XML一致性、入口拒绝执行、文档和路径检查。未执行新的PhysX、PPO或MuJoCo实验，因此没有新版工程门槛或策略通过结论。后续实现须重新完成相应容器工程验收、从零学习证明与独立行为评估。

当前停训状态以[交接](current_handoff.md)为准。重建方案见[计划](../demo_project/upright_rl_demo/docs/rebuild_plan.md)，不沿用历史自动授权发批，不恢复第26批、周期定时器或goal。

重建目录发布提交`85e7c94`，已同步demo索引和manifest；归档分支仍固定b357dd7。检查证据位于`runs/_upright_reorganization_20261030/{archive_manifest,reorganization_review}.json`。规格检查读得左右髋范围[-1,1.2]rad、轮目标±25rad/s及后轮力矩±0.38Nm；这些是源模型约束，不是学习结果。
