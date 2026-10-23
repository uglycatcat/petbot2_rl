# upright_rl_demo

PetBot2 的前轮抬起、后轮支撑、两轮遥控及前轮落地任务。训练控制目标为车架关节位置和左右后轮速度，头耳保持默认位置；不使用示范动作轨迹。模型来自主仓库只读的 `robot_description/`。

## 代码导览

| 入口 | 对应任务 | 用途 |
|---|---|---|
| [`train.py`](train.py) | [`upright/task.py`](upright/task.py) | 历史静止初态起立任务；保留已完成的静止基线及其检查点接口。 |
| [`train_moving.py`](train_moving.py) | [`upright/moving_task.py`](upright/moving_task.py) | 历史四轮行驶中接管和两轮遥控任务；保留旧实验复现入口。 |
| [`train_cycle.py`](train_cycle.py) | [`upright/continuous_task.py`](upright/continuous_task.py) | 当前完整运动循环：实际行驶中接管、前轮抬起、持续混合遥控、前轮落地、普通控制交接。 |

三种入口的观测、模式和检查点契约不同；检查点只按其所属任务代码和固定版本加载。`upright/{moving,cycle,continuous}.py`提供各代任务状态机，`continuous_task.py`复用旧循环的基础状态与控制接口。旧完整循环检查点须在其固定提交运行；新入口仅支持显式actor迁移或同契约重载。正式实验在 `demo_project/runs/` 中建立固定提交的独立副本；**本目录是可维护源码，不只存最新一版，也不直接存正式训练检查点**。

## 工具与产物

| 路径 | 职责 |
|---|---|
| `tools/launch_cycle.py`、`tools/evaluate_cycle.py`、`tools/watch_cycle_jobs.py` | 完整循环批次的固定版本部署、独立评估与进程监视；启动前核查已有状态，避免重复启动。 |
| `tools/run_guarded.py` | 宿主机显存守护与容器训练编排。 |
| `tools/watch_cycle_evaluation.py` | 单批评估进程退出事件监督；已有完成通知则不重复，异常退出时向原会话排队，禁止自行重启训练。 |
| `tools/probe_cycle_yaw.py` | 固定策略下的受控差速诊断，记录实测轮速和执行器力矩估计；诊断输出不作为任务成功评估。 |
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
