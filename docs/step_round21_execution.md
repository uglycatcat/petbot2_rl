# Step第21轮执行：规定短行后的停车反馈（2026-11-03）

第20轮两路完整均0/64；seed70合法落顶21、四轮上台0，seed71合法落顶60、归髋45、规定短行36，但轮请求始终≥11.65rad/s。任务记忆固定尺度保留，停车须同时协调轮速与默认髋。有限当前策略诊断：原样0完成、只归零轮请求0、轮请求和髋目标同时归零31；5例静止后的接触/距离失败保留，诊断不算策略通过。见[终局与诊断](step_round20_results.md)。

第21轮固定`a983e893b8864f9efaf71fd04290a1332e562c5f`已推送`demoproj/step_rl_demo`，代码摘要`f7717f57ad4a597d422263411c994e8a624bdecdefb5c68ed0d995bdf79695fb`。唯一配置变化为`post_drive_stop_cost=command_motion_v21`：在真实规定短行完成后的下一个控制步起，新增每秒`5*(轮请求/25)^2 + 2*(max(整机速度−0.02,0)/0.3)^2`代价。记忆随后不因支持/姿态丢失而关闭，每回合清零；此前所有阶段新增代价为0。原传感/任务记忆、对称动作、控制、物理、奖励其余项及完整验收保持，没有任何脚本制动或锁髋。lower_body世界俯仰目标及0.1→1课程、末200更新/独立评估完整代价保持。详见[实现](../demo_project/step_rl_demo/docs/round21.md)。

161项容器测试、10项受保护仿真通过。128环境4→8同次完整恢复、中途/终局独立重载、512环境4更新及重载有效；两份工程终局各16环境成功为[0, 0]，仅证明工程链路有效。两路真实策略动作、静止及有限制动动作共256条场景回放，物理/观测/原验收误差均0；标量新代价重算及奖励差误差≤2.66e-07。覆盖36例正式轨迹真实短行、支持丢失后代价保持、31例物理完成以及新回合清零；登台前无代价。Actor/critic检查点恢复、梯度与26维经验归一化/3维0–1记忆验证有效。所有物理守护均有效，无OOM/PhysX/Python错误；启动仿真前重启过一个等待器以改用已独立审查的诊断标记，未运行或中止训练。

正式`step_round21_20261103`于北京时间10:57启动，seed70/71在`petbot-step-gpu0/1`、仅物理GPU0/1，各512环境×500更新，终局各独立评估64环境。实际initial.pt为随机actor/critic、空优化器、无冻结参数；两个新run_id、不同种子初始权重、每路至少2次实际PPO更新及GPU归属已核验。没有历史训练检查点或诊断数据进入新PPO。源模型指纹`de1862d58f9989ae7eb0de5b20188e2ffc7052d2b9d5588446eef4b166b5f53f`保持，正式冻结源码/配置与工程门槛相同。

待正式验证：是否学会撤销驱动、保持默认髋并真正停稳；更早碰沿/落顶后再抬、世界俯仰过大和静止后的接触/距离条件仍须分别审查。新代价不直接修复所有早期缺陷，不能将31例外部动作诊断或短训通过视作新策略能力。不发布新桌面策略，未运行sim2sim，正式训练占位保持空。

本事件审查、有限诊断、实现、工程门槛及启动核验共约29分钟，正式长训另计。campaign enabled=true、active指向本轮；第20轮在实际新训练核验后标记handled，旧事件只补记接收。同一完成/异常事件链延续，无周期定时器、goal或子agent。主仓库manifest与demo分支同步，default.xml仍指同一demo分支；main发布后同步repo manifest checkout。

证据：工程目录`agent_review.json`、四项动作回放、`normalization_audit.json`、`checkpoint_audit.json`、`curriculum_audit.json`、`control_acceptance_audit.json`及guard；正式目录`revision.json`、`plan.json`、`startup_verification.json`、`actual_initial_checkpoint_audit.json`及各路origin/progress。
