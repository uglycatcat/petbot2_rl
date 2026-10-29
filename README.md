# upright_rl_demo

PetBot2 的前轮抬起、后轮支撑、两轮遥控及前轮落地任务。训练控制目标为车架关节位置和左右后轮速度，头耳保持默认位置；不使用示范动作轨迹。模型来自主仓库只读的 `robot_description/`。

## 从零训练（当前入口）

`train_scratch.py`从随机权重学习完整循环，不需要历史runs、检查点、冻结父网络或示范数据。配置为`configs/scratch_v2.json`：每份1000更新、单个20ms策略、全部参数可训练。起立奖励采用有限的新进展奖励，四轮停留不再持续赚取过渡与平稳奖励；成功条件保持。课程从静止起落到低速，再独立扩展共同髋初态、接管速度、双向直行、转向与弧线。旧入口仅保留历史评估/分析；旧继承式发批已禁用。

容器环境由`environment/Dockerfile`和固定依赖构建，基础镜像为`nvcr.io/nvidia/isaac-lab:3.0.0-beta2`；需按NVIDIA许可自行取得。当前镜像没有registry digest，版本及Isaac Lab源码摘要由`tools/environment_report.py`记录；新环境必须重过工程门槛，不能承诺跨GPU/驱动逐位一致。

在专用容器内、源码已提交后运行（模型目录只读挂载，源码路径保持相对）：

```bash
/isaac-sim/python.sh tools/environment_report.py
/isaac-sim/python.sh tools/prepare_asset.py --headless --device cuda:0
/isaac-sim/python.sh train_scratch.py --run-dir logs/my_run --seed 60 --headless
/isaac-sim/python.sh train_scratch.py --run-dir logs/my_run --evaluate logs/my_run/model_999.pt --output logs/my_run_eval --num_envs 64 --headless
```

最后一步同时输出物理轨迹、能力评估和单网络ONNX包，校验ONNX Runtime与PyTorch数值；网络导出不等于MuJoCo闭环通过。当前桌面默认C58保持独立。新网络部署必须使用相同20ms历史/决策与明确的状态机、2ms控制器。

同次中断恢复：`train_scratch.py --run-dir logs/my_run --resume logs/my_run/model_500.pt --headless`，读取原配置并继续到原1000更新，不另加1000；恢复网络、优化器、课程和随机状态，物理回合重新开始。跨运行初始化被拒绝，改配置的新实验重新随机初始化。`initial_state.pt`仅为审计初值，不能作为恢复检查点。

宿主机自动编排使用`python3 tools/launch_scratch.py PLAN.json`：固定干净提交、每份独立源码副本和源模型资源重建、GPU显存守护、训练/同次恢复、少量独立评估、完成或异常事件续接。计划只含种子/预算/场景，不接收父模型；正式计划必须引用同任务版本工程通过证据。队列结束仍须审查失败和物理轨迹。

## 历史代码导览

| 入口 | 对应任务 | 用途 |
|---|---|---|
| [`train.py`](train.py) | [`upright/task.py`](upright/task.py) | 历史静止初态起立任务；保留已完成的静止基线及其检查点接口。 |
| [`train_moving.py`](train_moving.py) | [`upright/moving_task.py`](upright/moving_task.py) | 历史四轮行驶中接管和两轮遥控任务；保留旧实验复现入口。 |
| [`train_cycle.py`](train_cycle.py) | [`upright/feedback_task.py`](upright/feedback_task.py) | 历史继承式完整运动循环：实际行驶中接管、前轮抬起、以轮轴地面航向反馈的持续遥控、前轮落地、普通控制交接。 |

三种入口的观测、模式和检查点契约不同；检查点只按其所属任务代码和固定版本加载。`upright/{moving,cycle,continuous}.py`提供各代任务状态机，`feedback_task.py`复用持续循环并添加转向反馈，`fast_feedback.py`提供20 ms或2 ms的有界转向内环，历史外层策略每60 ms更新，`balance_reflex.py`可在稳定两轮段增加20 ms的冻结车架/共同轮速反馈；`ground_motion.py`定义轮轴地面航向，`local_frame.py`可将观测中的世界水平向量与参考航向转换到机器人当前行驶方向坐标；机身前向投影另作历史诊断，不与新坐标口径混用。`balance_adapter.py`提供冻结完整循环actor上的有界车架/共同轮速及独立转向修正，只在稳定两轮阶段启用；含独立JIT导出，ONNX尚未实现。`continuous.py`可显式选择混合运动或瞬时容差跟踪奖励，独立验收口径不随奖励选择改变。旧完整循环检查点及对应checks须在其固定提交运行；历史入口仅用于其固定版本评估；当前训练使用无父依赖的train_scratch.py。正式实验在 `demo_project/runs/` 中建立固定提交的独立副本；**本目录是可维护源码，不只存最新一版，也不直接存正式训练检查点**。

## 工具与产物

| 路径 | 职责 |
|---|---|
| `tools/launch_cycle.py`、`tools/evaluate_cycle.py`、`tools/watch_cycle_jobs.py` | 完整循环批次的固定版本部署、独立评估与进程监视；启动前核查已有状态，避免重复启动。 |
| `tools/run_guarded.py` | 宿主机显存守护与容器训练编排。 |
| `tools/watch_cycle_evaluation.py` | 单批评估进程退出事件监督；已有完成通知则不重复，异常退出时向原会话排队，禁止自行重启训练。 |
| `tools/watch_existing_gate.py` | 为已经启动的工程验收进程接入一次性完成/异常续接；核对PID身份，不启动或重试仿真，进程正常退出仍需agent核查结果。 |
| `tools/evaluate_candidates.py` | 按固定计划只做候选验证；支持原检查点或未经PPO更新的显式父策略迁移，核对源码/检查点哈希并保留完成/异常事件，不启动训练。 |
| `tools/export_cycle_policy.py` | 从已完成的固定策略评估导出外层actor和冻结内部反馈、哈希及控制契约，并做CUDA重载一致性检查；网络包不包含完整控制器，不能视为MuJoCo闭环验证。 |
| `tools/probe_rise_authority.py` | 固定起立策略的车架/共同轮速修正幅度干预；仅修改内存中的幅度缓冲，记录裁剪前后动作，1/1对照须复现原轨迹，不作为独立任务验收。 |
| `tools/probe_landing_state.py` | 在原评估循环中记录复位前的车架关节、落地目标与地面保持计时；新增诊断必须先对照原轨迹，不能以增加记录替代任务验证。 |
| `tools/probe_cycle_yaw.py` | 固定策略下的受控差速诊断，记录实测轮速和执行器力矩估计；诊断输出不作为任务成功评估。 |
| `tools/probe_continuous_yaw.py`、`tools/analyze_continuous_response.py` | 第七版固定策略反馈干预及离线响应分析；探针须使用相应固定版本，窗口诊断不替代验收。 |
| `tools/probe_cycle_substeps.py`、`tools/probe_heading_reference.py` | 固定版本的物理子步/航向定义诊断；只读采样不得改变接触传感器历史时序，探针须配套其固定任务版本。 |
| `tools/probe_fast_feedback.py` | 固定第九版策略的反馈时序/增益诊断，baseline须复现原轨迹；诊断不改变检查点契约，也不替代正式评估。 |
| `tools/probe_policy_cadence.py`、`tools/probe_common_feedback.py`、`tools/probe_balance_frame.py` | 固定第十版actor的策略周期、前向反馈与车架变化率诊断；均单独保存诊断输出，不覆盖正式评估。 |
| `tools/probe_policy_distribution.py` | 固定第十一版策略在稳定两轮段的网络输出、动作分布和均值灵敏度诊断；只读钩子不采样新动作，仍须核对原轨迹。 |
| `tools/probe_yaw_bandwidth.py`、`tools/probe_frame_adapter.py` | 固定第十三版策略的转向反馈周期/增益及车架修正幅度诊断；在各自固定副本运行，基线需逐项复现原轨迹。 |
| `tools/analyze_cycle_quality.py`、`tools/check_resume_state.py` | 离线质量重算与续训状态核查。 |
| `tools/launch_moving.py`、`tools/evaluate_moving.py`及早期 `launch_round.py`、`evaluate_v2.py` | 历史任务工具，仅用于复现相应固定实验。 |
| `artifacts/` | 从只读源模型生成的本地派生资产，已忽略，不是唯一源码。 |
| `../runs/` | 所有 demo 的固定代码副本、检查点、日志、评估轨迹及状态；体量大且多数为历史证据，不按目录名推断当前实验。 |

## 按需阅读

- 当前任务、固定版本、运行状态入口和下一步：[主仓库当前交接](../../docs/current_handoff.md)。
- 已确认的控制目标与验收边界：[需求基线](../../docs/current_scope.md)；批次继承关系：[训练谱系](../../docs/upright_training_lineage_20261023.md)。
- 具体实验设计与结果：从[文档索引](../../docs/README.md)进入对应批次，勿把本 README 当实验进度记录。
- 项目物化、分支和容器配置：[demo 索引](../README.md)及主仓库[部署说明](../../README.md)。

训练只在专用容器内运行。源模型与正式运行副本保持不变；调整训练代码应另行提交新版本并单独评估。

### 稳定两轮阶段的固定反馈

`train_cycle.py`支持`--balance-reflex off|frozen20|parent_full20`。主策略、五帧观察历史与PPO采样始终60 ms；`frozen20`仅在两轮阶段1.8秒之后，每20 ms用冻结父修正器更新车架/共同轮速。冻结控制器作为环境内部反馈随actor检查点保存，不随PPO更新；实际执行保留绝对限位和整60 ms变化率边界。起立、落地和普通控制交接不启用该反馈。

`parent_full20`仅用于全范围任务，从指定v22父检查点建立完整确定性网络快照，保存为`actor.parent_balance`。每60ms记录“外层采样动作−冻结父均值”，随后20/40ms只刷新冻结父的当前物理帧并加回该偏移；外层PPO更新不会改变内环参数，转向仍沿用原2ms反馈。`checks/parent_feedback_contract.py`检查CUDA输出、采样偏移、冻结梯度和严格重载，`checks/full_range_runtime.py`检查真实PPO更新及两份内部模块冻结。该训练候选需独立工程和效果验收；现有MuJoCo导出器不支持此新合同，不能只导出外层网络用于部署。

部署必须同时保留外层actor和`actor.balance_reflex`（可独立`torch.jit.script`导出），以及相同历史缓存、阶段门控、目标约束和转向反馈；仅导出外层actor不能复现`frozen20`。固定父策略提频诊断入口`tools/probe_balance_cadence.py`只用于确定性评估，不能用作PPO训练入口；第十四批诊断复现使用95dded7固定副本。

### 稳定优先的完整循环

`--tracking-objective stability_first --tracking-weight 4`使用平滑实际运动和宽容差功能奖励，避免把不响应非零指令当作最优。完整循环增加分阶段的持续支撑丧失/提前落地/失控检测；`upright/stability.py`定义判据，`upright/functional.py`给出与历史严格精度分开的基本遥控评估。数值为版本化工程初值，不代表任意场景不会跌倒。

第十六版入口显式迁移验证过的第十五版B检查点，保留冻结内部控制器和位置/轮速接口；指令序列覆盖六类别的全部30种有向转换。评估可设置至180秒两轮保持，场地边界随允许速度与时长定义。旧检查点继续在对应固定副本复现，不能直接用新任务口径覆盖旧结果。

单进程环境数量上限为4096；8192及以上已实测触发PhysX的64K材质上限，不能因进程退出0或损失有限就采信结果。可使用`tools/run_distributed_guarded.py --gpus 1,2,3`启动一份同步PPO（每卡一个仿真实例、共享策略梯度），不能用多个独立策略冒称多卡加速；`run_guarded.py`会拒绝超限配置，并在日志出现PhysX错误时终止当前token所属任务。工程放行还检查短训/重载日志及检查点来源，拒绝无效短训派生的评估。

### 观测坐标对照

`train_cycle.py --observation-frame world|local|balance_local`选择观测编码。`local`在每次采样、历史堆叠及限幅之前旋转水平线速度/角速度、位置误差和参考航向；`balance_local`保留原起立输入，在进入两轮阶段1.2–1.8秒逐渐启用旋转，并保持至落地交接。身体重力、关节、命令、竖直速度、时钟及原有动作接口不变。检查点契约记录选择，禁止跨编码直接重载；只能按入口规定显式迁移父检查点。旧批次继续使用固定版本。

### 平稳落地后的车架对齐

`--handoff-alignment before|after`分别选择旧交接顺序和分步交接。`after`仍要求连续0.3秒四轮接触、无机身接触及平稳姿态/速度，再由普通控制按既有位置/轮速变化率调整车架。最终完成仍要求车架距目标小于0.1 rad且稳定保持1秒；没有把落地或进入普通控制直接计作完成。检查点记录该契约，旧版本不能直接重载，必须显式迁移已验证父策略。

### 起立阶段修正

`--refinement-target rise`在完整父策略外增加有界车架位置/共同轮速修正，仅起立阶段启用，并在进入两轮后的0.6秒内退出。原actor、两轮/转向修正、观测统计和内部反馈全部冻结；左右差速仍由继承策略处理，普通控制与落地完成判据保持。初始化为零修正，工程检查须证明确定性父输出一致、阶段隔离和短训只更新新修正头。起立阶段增加探索，稳定两轮段仍沿用父分布。

训练可缩短两轮保持以提高起立样本占比；这不替代180秒独立遥控和落地验收。`checks/rise_transfer.py`与`checks/rise_runtime.py`分别检查CUDA参数/概率/阶段契约和真实短训冻结状态，旧任务仍在各自固定副本复现。

### 线速度继承与航向保持

`--transition-contract linear_heading --refinement-target all`启用修正版完整循环：起立/落地捕获实际前向速度和航向，暂存遥控新命令；2ms差速反馈抑制过渡偏航，过渡期间关闭学习差速残差，两轮段恢复遥控及策略差速修正。不会清零物理速度，允许为平衡调节共同轮速。`all`更新外层actor全部分支及critic，20ms内部反馈仍冻结；旧`rise`仅更新起立头。

落地判别通过body IMU重力方向和实际车架编码器计算轮轴几何，按后轮支撑平地的任务前提判断前轮回到地面；不需要轮接触传感器。前轮几何到位且姿态角速度适宜，保持0.12秒即交给普通控制对齐车架，最终保持1秒确认交接。取消旧4秒落地失败门槛，未完成仍记录为未完成，真实失稳和整回合期限继续有效。仿真接触数据只可用于训练奖励/独立诊断，不能当作硬件传感器；本实现的世界位置/速度仍来自仿真，实机估计器尚未实现。

新版`balance_local`起立观测以接管航向为坐标锚；各旧固定版本保留原行为。速度范围必须显式指定，源轮速上限推导的运动学极限、普通控制器的软件上限和实测硬件峰值不能混称。GPU验收、未通过场景和部署状态见主仓库当前交接页。

### 全范围独立调用

`--task-profile full_range --transition-contract linear_heading --refinement-target all`选择新版独立调用任务。后轮目标仍限±25 rad/s；线速度/转向请求投影到左右轮联合可达域，起落使用捕获航向和速度，两轮段恢复遥控。曾离地后意外回落或真实失稳结束本次调用，单独记录中断；训练重置不代表部署自动再次起立。

`--balance-noise-factor 1|4|16`控制稳定两轮探索，`--actor-normalization rebase_recent`在显式任务迁移时按父critic统计重参数化，须验证确定性输出不变及内部反馈冻结。评估保存`behavior`分阶段运动/关节/力矩及最大过渡偏航，汇总成功率不能替代行为审查。新旧合同禁止直接混用，旧策略在各自固定版本运行。

### 全范围训练分层采样

`train_cycle.py --task-profile full_range --training-mixture stratified_v1`将训练环境等分为低范围保留、中等运动、较高运动、低运动全车架、全范围五组。分组在环境生命周期固定，避免困难回合较短而丢失物理步覆盖；一次调用内幅度分组不变。该参数只影响训练抽样，`--evaluate`强制关闭分层并使用显式评估范围，外部遥控和源电机限制保持不变。`off`为相同任务的原全范围抽样对照。日志`Stratum0`–`Stratum4`记录各层起立/完成比例，配置记录实际启用状态；正式批次必须先过容器短训及关闭分层后的重载检查。

### 固定策略过渡速度诊断

`tools/probe_transition_speed.py --transition-speed-probe identity|rise_brake|landing_brake`仅接受`--evaluate --checkpoint`及原检查点对应的任务参数。`identity`保留控制；另外两种模式只在起立或落地阶段，以既有0.6m/s²变化率将捕获的速度参考收至零，不改变物理速度、接管条件、两轮指令、航向、动作限制或失败判据。合同明确记录开发干预，轨迹增加过渡参考与轮轴几何；必须核对identity逐帧复现、成功/失败及超时，不能把诊断当训练或部署通过。

### 起立速度参考学习对照

v24的`--balance-reflex parent_full20 --rise-speed-reference capture|brake`将起立参考作为训练和重载合同的一部分。`capture`保留接管实测速度；`brake`只在起立阶段按0.6m/s²把参考移向零。两轮和落地指令、阶段与失败条件、源执行器约束保持，不能把参考变化称为实际速度清零或稳定性通过。原固定策略诊断使用其5d7fe57副本，新任务不能跨合同直接重载。

从验证过的v23检查点显式迁移时，完整恢复外层actor和critic，保留检查点内原有的完整父反馈网络，不重新拍快照；近期统计重参数化仅改变外层等价坐标，optimizer重新建立。合同、源SHA和实际保留状态写入config；`transition_reference_speed`记录每周期真实参考。工程短训、严格重载和后续完整保持评估必须分别核查，起立入口增加不能代替早期两轮稳定。


v25能力训练沿用`train_cycle.py`，在完整范围配置上添加`--capability-version v25`。
`--mirror-training`启用配对命令与镜像均值损失；`--adaptive-curriculum`启用两条表现课程。
正式默认1000更新；版本、父检查点和课程开关必须显式固定。恢复时使用同契约`--checkpoint`，
从下一个更新继续并开启新物理回合。独立评估加`--evaluation-pattern paired_speed|paired_arcs|rise_grid`可检查能力边界。
准备是phase1中尚未离地的连续策略控制段，无强制停车/归零；落地共同髋角回默认0，其余交接语义沿用。
有限批队列`tools/launch_cycle.py`支持计划项`sequential_gpu_waves: true`，每GPU最多一份PPO，
失败冻结后续任务供事件恢复；工程门槛及固定提交要求保持。
