# PetBot upright：独立从零训练

四轮运动接管 → 直线起立 → 两轮遥控 → 前轮落地 → 普通控制。Isaac Lab + RSL-RL PPO，单Actor–Critic；新实验随机初始化，不加载历史权重/教师。只有同次受迫中断可恢复完整状态。

## 入口


| 入口 | 用途 |
|---|---|
| `python3 project.py check` | 只读核对用户要求和源MJCF约束 |
| `python3 train.py --dry-run` | 查看配置和源码/模型身份；不导入仿真器、不创建训练产物 |
| `train.py` | 唯一PPO训练入口；新运行随机初始化，`--resume-interrupted`只接受同次中断 |
| `python -m upright.prepare` | 在专用容器内由只读模型生成派生资产；仅在已授权专用GPU2/3容器执行 |
| `evaluate.py` | 显式指定新运行和检查点的固定策略验证，不训练 |
| `export_policy.py` | 新策略单网络ONNX及manifest，不自动更新桌面包 |

基础实现见[第1轮说明](docs/round01_implementation.md)，原版执行见[第20轮执行](../../docs/upright_restart_round20_execution.md)，当前工作见[回退与停车诊断](../../docs/upright_round20_rollback_landing.md)。用户需求见[需求基线](../../docs/current_scope.md)，历史来源见[源码取用](docs/source_provenance.md)和[封存说明](ARCHIVE.md)。没有旧训练入口、自动寻找latest模型或历史目录回退路径。

## 配置与执行

当前候选为`configs/round21r5.json`，见[21r5设计](docs/round21r5.md)。21r4落地后退减小但低速/零令能力保留不足；本次重分配现有20%回归，增加静止和最低运动课程，保持80%当前范围、控制器与奖励。只有工程验收后才从零发批。原配置用于固定版本追溯，不能混用权重和控制契约。

demo每份1000更新、2048环境，物理GPU2/3内按明确计划运行；正式交付预算可提高。训练从默认起立、静止循环逐步扩展到移动/初态课程，同一次训练内按门槛晋级并保留低难度回归。双向连续0–0.75m/s是目标，不代表已有能力；普通四轮全能力不随课程缩小。源模型只读，轮目标±25rad/s，实测超调单列。

每次实验固定源码、模型、配置指纹和run_id；检查点保存优化器、归一化、课程、随机状态和更新计数。恢复只接受同次最新完整状态，物理在途回合明确复位，不重复累计预算。输出放`outputs/<明确运行名>/`，不扫描历史runs或latest模型。

先做容器内逻辑检查及相应PhysX/短训/恢复/重载门槛，再固定提交发批。工程通过不等于策略通过；评估需读阶段、请求/执行/实测以及成功/失败轨迹。CPU检查不能替代仿真。依赖见[requirements](environment/requirements.txt)，宿主只编排，不执行Isaac训练。

## 事件驱动运行


`tools/launch_round01.py <明确计划.json>`在干净固定提交和已接受工程指纹下启动两份随机实验；宿主只编排，PPO与评估在专用容器内执行。计划必须指定物理GPU2/3、种子、1000更新、环境数、提交、config配置路径、execution_document执行文档和工程证据，不能搜索历史模型。固定运行副本不随开发代码修改。

启动器保存launch_status/eval_status/eval_finished、日志、退出原因与实际运行PID。独立验证按起立→静止循环→低速混合展开；完全未进入两轮时跳过后续；若已训练更高阶段且低速物理完整至少26/32，再增加当前能力阶段验证，其余显式记为跳过。显存受迫中断只从同一次运行最新完整检查点恢复，代码错误不自动重跑。完成与异常发送一次性事件到已授权原会话；PID退出观察器负责通知缺失检查，不设置周期定时器。

每个控制周期在策略观测前调用`Controller.update_motion_reference()`一次；观测用`motion_command()`，日志分别记录`requested_command`、`control_reference`、`effective_command`及实际关节目标和机体运动。普通控制按源范围执行，不受两轮加速度限制。

## 文档与历史

实时状态仅维护在[当前交接](../../docs/current_handoff.md)和[21r6执行](../../docs/upright_restart_round21r6_execution.md)。重启系列历程见[按需索引](../../docs/upright_restart_history.md)，旧系列文档/运行产物在服务器`history/upright/`，不进Git，见[归档说明](../../docs/upright_history.md)。新工程运行不依赖这些归档。README不再逐轮追加结果。
