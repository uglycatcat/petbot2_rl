# Step第20轮执行：任务记忆固定尺度（2026-11-03）

第19轮两路完整均0/64，合法前轮落顶4/51、四轮上台与归髋0/41、规定短行均0。真实回放和代价有效；seed71任务记忆跳变与归位破坏有关。固定当前策略的单位尺度干预使规定短行0→36，但仍不停车；详情见[终局与诊断](step_round19_results.md)。

第20轮固定`a6b85a9c7987bb4cb4fa4a1f0620041a3deffe3f`已推送`demoproj/step_rl_demo`，代码摘要`b6feb6d41c193789eff71d4992249fcfc2e00e757be69817b4f3bb0108ee19f4`。唯一配置变化为`observation_normalization=sensor_memory_unit_v20`：Actor和critic前26项继续原经验归一化，末3项传感任务记忆保持原0–1值；保留原生RSL-RL MLP/Beta/PPO。原传感记忆计算、29维观测内容、动作/物理/奖励和验收全部保持，没有输入屏蔽、锁髋或脚本停车。世界俯仰训练代价0.1→1课程保持，末200更新和独立评估均为完整代价。

155项容器测试、9项受保护仿真通过。128环境4→8同次完整恢复、中途与终局独立重载、512环境4更新及重载有效；新Actor/critic类、统计、梯度与检查点重载均核对。工程两份终局各16环境成功数为[0, 0]，只证明工程有效。两路128条真实动作及64环境静止回放物理/原验收完全一致，完整俯仰奖励一致，真实观测的26维归一化前缀及3维固定尺度后缀逐项验证；47/4个首回合任务记忆启动被覆盖。工程检查无无效物理、Python异常或OOM。

正式`step_round20_20261103`于北京时间06:34启动，seed70/71在`petbot-step-gpu0/1`、仅物理GPU0/1，各512环境×500更新，终局各独立评估64环境。实际initial.pt为随机actor/critic、空优化器、无冻结参数；两个新run_id、不同种子初始权重、PPO至少2次真实更新及每路唯一GPU训练进程已核验。相同种子可复现相同随机初值，不等于载入旧权重；本轮没有任何历史训练检查点或诊断数据输入。源模型指纹`de1862d58f9989ae7eb0de5b20188e2ffc7052d2b9d5588446eef4b166b5f53f`保持，正式冻结源码/配置与工程门槛一致。

待验证：固定尺度从零学习是否能维持归髋并完成短行后真实停车；更早的碰沿、落顶反弹与过大世界pitch仍需审查。每轮须看请求/执行/实测、接触、各阶段进展和完整成功/失败轨迹；不能用短行36次的旧策略干预或短训通过替代正式新策略验收。不发布新桌面策略，未运行sim2sim，正式训练占位保持空。

本事件审查、有限诊断、实现、工程门槛及启动核验共约20分钟，正式长训另计。

同一campaign enabled=true、active指向本轮；第19轮agent_review已在实际新训练核验后handled。完成/异常继续原事件链，无周期定时器/goal。源码与主仓库manifest同步发布，default.xml分支未变；main发布后同步repo manifest。

证据：工程目录`agent_review.json`、`normalization_audit.json`、`checkpoint_audit.json`、`curriculum_audit.json`、guard及回放；正式目录`revision.json`、`plan.json`、`startup_verification.json`、`actual_initial_checkpoint_audit.json`与各路origin/progress。
