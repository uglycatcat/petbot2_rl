# PetBot STEP v2

独立训练机器人直行登上2cm平台：前轮只抬一次且不撞边，后轮滚上，恢复默认髋角、沿原线短行后停稳。输出左右共同髋角和共同后轮速度；地形信息仅开始传入一次。

**当前暂停，未经用户明确重新授权不启动训练或工程实验。** 最新状态与结果只维护在[STEP交接](../../docs/step_current.md)，完整目标见[STEP需求](../../docs/step_scope.md)。现有训练配置为512×500，尚不满足用户新要求的至少2048×1000；本次仅整理文档。

| 入口 | 职责 |
|---|---|
| [实现说明](docs/architecture.md) | 当前任务、观测、控制、奖励及验收；实现选择不等于用户硬性要求 |
| [管理流程](docs/automation.md) | repo、冻结、状态文件、分析和暂停 |
| `step_demo/task.py` | 唯一任务状态与奖励 |
| `step_demo/isaac_env.py` / `estimator.py` | PhysX、执行器及自身传感估计 |
| `step_demo/ppo.py` | RSL-RL PPO、GRU及同次恢复 |
| `step_demo/evaluation.py` / `review.py` | 原始2ms物理轨迹的独立复核 |
| `configs/step_v1.json` / `train*.json` | 当前场景/验收参数及训练配置 |

项目由repo物化，代码在`demoproj/step_rl_demo`，主仓库只管理索引和manifest。模型默认相对路径`../../robot_description`，只读。专用容器`petbot-step-gpu0/1`对应物理GPU0/1；已有容器不重复创建。`std_train_step_project`仍为空，sim2sim旧预览不是v2策略。

不启动仿真的静态检查：`python3 -m step_demo check`。开发/工程入口见`tools/`与`python3 -m step_demo --help`；命令存在不构成执行授权。历史文档与旧训练产物已移出当前入口，见[history定位](../../docs/step_history.md)。
