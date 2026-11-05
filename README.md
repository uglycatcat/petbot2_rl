# PetBot STEP v2

独立训练机器人直行登上2cm平台：前轮只抬一次且不撞边，后轮滚上，恢复默认髋角、沿原线短行后停稳。输出左右共同髋角和共同后轮速度；地形信息仅开始传入一次。

**用户明确尚未授权启动训练；当前已准备参数、双卡四路编排与录像功能。** 最新状态与结果只维护在[STEP交接](../../docs/step_current.md)，完整目标见[STEP需求](../../docs/step_scope.md)。正式候选为每路4096环境×1000更新、128步采样；两组初始髋探索强度，各两个种子。

| 入口 | 职责 |
|---|---|
| [实现说明](docs/architecture.md) | 当前任务、观测、控制、奖励及验收；实现选择不等于用户硬性要求 |
| [管理流程](docs/automation.md) | repo、冻结、状态文件、分析和暂停 |
| `step_demo/task.py` | 唯一任务状态与奖励 |
| `step_demo/isaac_env.py` / `estimator.py` | PhysX、执行器及自身传感估计 |
| `step_demo/ppo.py` | RSL-RL PPO、GRU及同次恢复 |
| `step_demo/evaluation.py` / `review.py` | 原始2ms物理轨迹的独立复核 |
| `configs/step_v1.json` / `train*.json` | 当前场景/验收参数及训练配置 |

项目由repo物化，代码在`demoproj/step_rl_demo`，主仓库只管理索引和manifest。模型默认相对路径`../../robot_description`，只读。当前部署使用GPU2/3，每卡两个专用容器，各仅暴露所在的一张物理GPU。GPU2为A/B seed70，GPU3为A/B seed71；第二路容器名加`-b`。计划模板见`configs/batch_dual_gpu.json`，须发布冻结并获得启动授权后使用。GPU0/1原容器保持空闲。`std_train_step_project`仍为空，sim2sim旧预览不是v2策略。

不启动仿真的静态检查：`python3 -m step_demo check`。开发/工程入口见`tools/`与`python3 -m step_demo --help`；命令存在不构成执行授权。历史文档与旧训练产物已移出当前入口，见[history定位](../../docs/step_history.md)。

正式批次结束后自动选择本批已评估最佳策略，输出`artifacts/step/videos/<批次名>/best.mp4`及来源说明；控制文件`video_root`可改保存位置。成功与失败均如实显示，录像失败不会重训。手动录制同次检查点可用`evaluate --eval-envs 1 --video`，其余参数沿用该固定运行副本；不启动PPO训练。
