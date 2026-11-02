# Step第18轮执行：世界俯仰软代价（2026-11-02）

第17轮两个种子均0/64完整，合法前轮落顶48/48，四轮上台47/0、归髋43/0、规定短行0/0；56/59例在落顶后再次离地，seed70另有6例躯体倾斜超界。两路2ms实际动作回放与原轨迹完全一致，世界pitch约−66°至+51°，变化率峰14.880/14.775rad/s。见[整批终局](step_round17_results.md)。

固定`ddafc026f281e3dd2582004829958b329250bebd`已推送`demoproj/step_rl_demo`；运行代码摘要`859882870f40043628998049db17525f7a308a19cfa41d3e83698b919c90bfb6`。`pitch_v18`每2ms积分`0.1·pitch_rate² + 2·max(|pitch|−0.35,0)²`，针对IMU所在lower_body的世界姿态。0.35rad和权重是工程初值，不是用户硬阈值；必要的髋发力和适度姿态变化仍允许。新增真值/IMU估计角度及变化率日志，29维传感观测、单次抬轮、对称控制、源模型、原归髋/沿线0.15m/实际停稳0.5s验收保持。

141项容器测试及8项受保护仿真通过，无物理/Python错误或OOM。128环境4→8完整恢复、512环境4更新和独立重载有效，优化器步数及actor/critic一致；两份短训各0/16完整，均前轮撞沿，只有工程意义。128条真实动作回放与64环境静止1s证明物理状态、终止和验收保持，独立pitch状态误差小于1.3e-6、代价累计误差小于7e-8、奖励差值误差小于2.5e-7。污染新增真值/诊断字段不会改变actor输入。原动作只用于有限工程诊断，没有加载历史权重、进入PPO或变成示范数据。

本轮22:11（北京时间）启动：种子70/71分别在`petbot-step-gpu0/1`使用物理GPU0/1，各512环境×500更新，终局各独立评估64环境。真实initial.pt、不同随机actor/critic、空优化器、无冻结参数、实际PPO更新和每路唯一GPU训练进程均已核验，容器只暴露对应单卡。正式冻结源码与工程运行代码相同；源模型指纹`de1862d58f9989ae7eb0de5b20188e2ffc7052d2b9d5588446eef4b166b5f53f`保持。

新代价是否能同时减少摆动、保留抬轮并完成停车仍待结果。下一次事件必须审查阶段与真实离地次数、世界pitch角度/变化率/阶段极值、传感提示与真实短行的偏差、髋及轮请求/执行/实测、接触和停车保持；检查不动、不抬轮或提前停车退化。世界姿态均值改善不能替代完整成功；实测轮速超调继续单列。

campaign enabled=true、active指向本轮；第17轮事件已handled，旧事件只补记接收。沿用同一单路/整批完成或异常事件链，无周期定时器/goal；本次未运行sim2sim，桌面策略不更新，正式训练占位保持空。`default.xml`仍指向同一demo分支，无新增/删改项目；demo manifest更新已核对提交，主仓库提交后同步repo manifest checkout。

证据：`demo_project/runs/step_round18_gate_20261102/agent_review.json`、`checkpoint_audit.json`、`control_acceptance_audit.json`、各回放result/guard、`push_verification.json`；正式目录`revision.json`、`plan.json`、`startup_verification.json`、`actual_initial_checkpoint_audit.json`、各路origin/initialization/progress。本事件分析/实现/工程/启动约15分钟，长训另计。
