# 当前交接：PetBot 完整运动循环起立

更新：2026-10-24。入口只保留当前决策与证据；运行状态以实际进程、日志和状态文件为准。

## 当前结论与工作

第七批已于2026-10-24 04:44:58完整结束：12组均跑至1199并正常退出，60/60评估有效，无缺失/失败，实际训练/评估进程已退出。原生交接742/768、30秒独立场景749/768，但两者联合跟踪及新版完整质量合格均0。该批修复了实际运动接管和持续混合换令覆盖，长时存活改善，遥控精度仍未达标。固定提交`54c2623`，见[第七批终局](upright_cycle_batch7_results.md)；不重复启动或无依据延长本批。

**第八批已通过GPU工程验收，于2026-10-24 06:10:41开始分批部署。** 固定提交`3121acf`，六对种子，A/B比较fine与标准差约减半的precision探索；均从第七批同种子A组1199迁移actor，critic/optimizer新建。转向反馈使用受控诊断支持的Kp2/Ki4/半差限速4，策略保留车架/共同轮速控制并提供小幅转向修正。设计、初始化、资源和评估安排见[第八版](upright_cycle_v8.md)。实际状态查`runs/_upright_cycle_batch8/launch_status.json`，不能重复启动。

四组反馈及三组共同轮速滤波诊断均已结束；修正版探针干预前轨迹最大差0。转向反馈改善低频误差，共同轮速低通反而恶化前向误差，因此不加入。所有探针仍不是正式训练效果，窗口诊断不能替代瞬时验收。

第七批原始证据在`runs/_upright_cycle_batch7/{launch_status,eval_status,eval_finished,quality_review,continuous_response_review,scalars_final}.json`及逐项config/trace/evaluation。原训练标量全部有限，60项评估固定版本和fingerprint一致。剔除换令后0.6秒，120 ms平滑转向误差仍约0.024 rad/s；单纯改统计窗口不能宣称达标。新结构已通过两组2048×20 GPU短训及四项重载：各交接63/64，瞬时联合跟踪仍0；工程通过不表示遥控达标，证据在`runs/_cycle8_gate_candidate1/release_gate.json`。

## 需求与约束

仅upright为活跃训练线；step未开始，recovery将重做。四轮运动中接管，不以预先停车为条件；允许起立时为平衡减速。后轮支撑阶段由同一策略持续执行遥控；四轮模式请求后使前轮落地并平顺交接。输出对称车架位置与左右后轮速度，头耳固定。见[需求基线](current_scope.md)。仿真交接仍使用内部普通行驶代理，外部控制器、MuJoCo策略闭环和实机尚未验证。

旧移动第五轮及以后夜间结果已废弃，不自动恢复第十轮；编号与父检查点见[谱系](upright_training_lineage_20261023.md)。第七版从第六版长训A组2399迁移actor；第八版从第七批A组1199迁移，单帧观察57→58，两版critic/optimizer均新建，不能称完整续训。静止默认初态起立已有阶段成果，不能替代完整运动循环验收。

训练只在专用容器运行，固定副本不得热改；源模型只读，不重启AGILE、不影响其他用户。发正式训练前完成资源实测，目标四卡计算利用率尽量接近80%，整卡显存78%守护。

## 自动续接

每日04/10/16/22定时器已取消：`runs/_recurring_upright/config.json`为`enabled=false`，用户systemd timer inactive；未经再次明确要求不得启用，也未创建goal。用户继续授权批次完成/异常事件向本会话排队；事件只通知agent先核查再决策，不自动重启训练。

第七批已正常发送完成通知，退出监督未重复发送，相关进程已结束。第八批部署/训练异常监督已附加；部署器在12组全部进入PPO后自动附加评估队列和评估退出监督。核查`launch_status.json`中的`eval_pid`及对应`eval_watch.log`，不能把部署器开始运行当成评估队列已经附加。

## 接手与证据

1. 先读根目录[AGENTS.md](../AGENTS.md)硬规则和本页，按需下钻，不遍历全部docs或runs。
2. 先核对`_upright_cycle_batch8/launch_status.json`、实际PID、训练日志及退出文件；如中断，诊断并恢复本批，不重开整批。
3. 第八批计划12组、四卡1/3/4/4分配、1200更新、60项评估；全部PPO启动后有限资源采样写`deployed_resource_samples.json`。200/600/1199及终局30秒独立种子20261026结果决定下一步，历史批次已收官。
4. 按同口径区分实际接管、长时两轮平衡、遥控分轴/联合跟踪、前轮落地、交接与完整合格。记录跨类别切换和瞬态/稳态误差；滤波速度辅助分析不能替代瞬时速度严格验收。

| 问题 | 阅读入口 |
|---|---|
| 当前设计、验收、部署 | [第八版](upright_cycle_v8.md)、本页对应状态文件 |
| 上一批终局和为何改结构 | [第七批终局](upright_cycle_batch7_results.md)、[第八版诊断](upright_cycle_v8.md) |
| 控制契约与工程边界 | [需求](current_scope.md)、[工程约定](engineering_handoff.md) |
| 源码和历史编号 | [demo导览](../demo_project/upright_rl_demo/README.md)、[谱系](upright_training_lineage_20261023.md) |
| 其他历史证据 | [文档索引](README.md)，按问题读取 |
