# 10月20—21日变更接手审查

范围：demo `4c1bd23..8f65e3f`全部8个变动文件；主仓库`b8a64cf..65ed585`。未修改训练代码、未重训、未删除产物。第四至九轮96次评估退出正常仅证明流程结束，不证明实验设计正确。

## 已确认的任务与实验问题

1. **接触丢失奖励方向反了（严重）**。`train_moving.py:62`将非负contact_loss_weight直接设为奖励权重；`moving.py`事件输出正1，`moving_task.py`除以dt后由奖励管理器乘dt。权重2时每次丢接触得到+2。第八轮B与第九轮全部组的固定运行配置实际启用了2；不能引用为“接触中断惩罚”的实验。CUDA事件复现为1，对应贡献+1.99999988。新增契约测试只断言事件为1，没有检查最终奖励符号。
2. **第五轮对照与评估分布改变**。旧默认模式是env_id%6，新mode_pattern(turn_share=0)只保留0/1/4/5，排除纯转向。第五轮终局原始takeovers核对：A每64环境为[16,16,0,0,16,16]，B为[8,8,16,16,8,8]。A不是原分布继续训练，A/B评估也不是同一场景集；总成功数差不能归因于奖励。所谓common只统一速度范围，没有统一模式分布。
3. **支撑过滤扩大了容忍范围**。过滤整个支撑合取条件，而非后轮接触；CUDA实测当前other=True时support仍True。
4. **支撑历史跨episode残留**。reset没有清空support_hist。CUDA实测稳定支撑后reset，active=False但下一次update的support=True。影响support_window>1版本；默认1的第四至九轮不受此问题影响。
5. **“只改测量”重测改变了策略输入**。support影响support_run，而support_run在205维观测内；过滤还影响lifted、阶段相关反馈。即使权重相同，也不能称为相同策略轨迹的离线重新计分。52/64须视为改变观测语义/判据后的闭环结果，不能视作能力直接提升。
6. **检查点契约遗漏新增参数**。contract没有support_window、contact_loss_weight、window_progress_weight，fingerprint只能检测代码变化，无法检测同代码下这些运行参数变化。allow-contract-drift还可用于非评估路径，未限制为独立审查用途。观测维度相同并不代表语义相同。初始化来源白名单也不是“任一终局可用”：当前表没有实际第九轮680770d提交。

## 已确认的工具风险

- campaign.analyze内覆盖name变量，真实第九轮内存重算的report.batch为frame_left，非moving_batch9；后续decision输出目录会错误。复现时替换save为空操作，未覆盖原结果。
- complete用len(status)>=16，而非逐任务检查status、退出码、检查点及结果；expected固定16，不适配第十轮两组。历史96次成功是本次逐项核实所得，不能依赖该自动判定。
- gate复用已有smoke/reload产物时没有校验来源commit、参数及检查点哈希；all_finite只是检查文本无nan且出现Learning iteration，未检查inf、全部标量或足额iteration。最终只硬查lifted>=64，未把提前退出/unfinished等作为拒绝条件。code_matches只检查upright与train_moving，不包含检查及编排工具。
- launch未建立独立启动锁，且未扫描其他批次实际进程；drive有锁不能覆盖直接launch。资源检查只预留一次4000MiB/卡，不按该卡计划并发累加；末端守护仍保留，但不能代替启动预算。
- common评估和retry未完整传递support_window等新参数；retry把native范围硬编码为当前小范围，且跳过遗留retry_running，无完整断点恢复逻辑。当前旧轮support_window=1不因此失效，未来使用3会导致口径错配。

## 未发现改动及结论边界

这段版本差异没有修改动作映射实现、网络结构、物理参数文件和源robot_description模型；观测拼接代码未改，但上述内部计时语义被改变。main的default.xml仅将ours传输改为SSH，双端需要SSH凭据；不是训练语义变动。demo manifest仍需后续按最终采用版本同步。

奖励/分布/过滤错误足以否定相关因果结论，但不足以宣布所有轨迹与权重毫无价值。第四轮固定版本及旧口径可作为审查基线；第五至九轮保留为待统一重评的候选权重，不作无缺陷对照证据；第十轮已中止，不补跑旧方案。下一步先修符号、分布、复位、契约和验收，再用同一场景集重评少量旧候选，决定是否从既有权重或新初始化训练。源模型、车架位置/轮速控制目标不变。
