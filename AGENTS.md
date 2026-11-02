# Step MuJoCo preview

用户于2026-11-02授权本独立项目：加载已审查策略ONNX，启动静止，Space执行上台阶策略，R完整重置。此授权替代此前step不做sim2sim的范围限制；训练仍在step demo，正式训练目录暂空。

- 本项目通过repo和`rlproj/std_s2s_step_project`管理；修改前核对远端，完整实现提交推送。根清单、索引和manifest同步，根共享改动不擅自提交。
- 源`robot_description/`只读；运行时只组合场景，不改机器人几何、质量、惯量、关节或执行器参数。
- MuJoCo运行时只有NumPy、ONNX Runtime与MuJoCo依赖，不从demo导入训练包或加载PT。导出/对齐工具可在训练容器CPU运行。
- 观测与共同髋/共同轮速输出遵循选定策略契约。真值只用于诊断/结束判定，不能注入actor；不把规则停车或手写抬轮当成策略效果。
- 每个发布策略记录来源commit/run/checkpoint/hash与选取依据。网络等价和程序可运行不等于动作通过；实际失败如实记录。
- 不影响固定训练副本或其它项目；仅GPU0/1，CPU推理/软件渲染优先，不创建子agent、goal或周期任务。
