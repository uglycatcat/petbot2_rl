# PetBot Step RL Demo

训练一个独立策略，让默认姿态的机器人直行、前轮抬起落到2cm平台、后轮滚上，然后归髋、沿原线短行并停稳。两维输出分别控制左右共同髋角和后轮目标速度，头部及耳朵保持默认位置。

**当前：v2首批完整成功0/128，按用户要求修正奖励和评估报告。** 新奖励区分抬高、有效越沿与真实落顶，停稳判据保持；见[本次修改](docs/v2_reward_update.md)。当前实现说明见[架构](docs/architecture.md)，旧批次产物由其固定源码追溯，不加载到新训练。

## 固定场景与验收

- 平台高2cm，前沿位于初始整机质心前30cm，边沿与行驶方向90度；速度由策略决定。
- 前轮只能离地一次，不能碰立面/棱边或落回台下；后轮保持支撑。
- 四轮上台、共同髋回默认、在原线上前进15cm后，实际速度不超过0.02m/s、偏航率不超过0.05rad/s，连续保持0.5s并再保持2s。
- 地形只在开始传入一次。GRU策略接收IMU、编码器、运动估计、上次执行目标及本地时钟；任务真值仅用于判定和critic。
- 源机器人及物理参数只读。所有新训练随机初始化，不加载旧权重、辅助权重、教师或轨迹。

## 代码与数据

| 路径 | 职责 |
|---|---|
| `step_demo/task.py` | 唯一任务状态、物理事件、阶段与有界回报 |
| `step_demo/isaac_env.py` | Isaac/PhysX、执行器、传感器与任务适配 |
| `step_demo/estimator.py` | 仅IMU与编码器的递推估计 |
| `step_demo/ppo.py` | RSL-RL PPO、GRU actor、非对称critic、同次恢复 |
| `step_demo/evaluation.py` / `review.py` | 独立重算2ms物理轨迹，核对完整流程 |
| `configs/step_v1.json` | 固定用户场景与验收数值 |
| `configs/train.json` / `engineering.json` | 唯一v2训练配置；工程为512环境、8更新 |
| `tools/validate_physics.py` | 有限电机正负轨迹；不进入PPO |

项目由repo物化，分支为`demoproj/step_rl_demo`。运行产物放主仓库忽略的`demo_project/runs/`，正式运行使用固定源码与资产副本。旧配置不兼容v2，旧运行必须用原固定副本追溯，不能迁移权重。

## 检查与工程入口

不启动仿真的模型/场景检查：

```bash
python3 -m step_demo check
```

容器CPU检查：

```bash
docker exec -e CUDA_VISIBLE_DEVICES= -w /workspace/demo_project/step_rl_demo \
  petbot-step-gpu1 /isaac-sim/python.sh -m unittest discover -s tests -v
```

仅允许本项目容器`petbot-step-gpu0`和`petbot-step-gpu1`，对应物理GPU0/1。已存在容器不要重复创建。默认相对路径读取只读的`../../robot_description`，也可显式传`--model-dir`。

提交后的源码可以冻结到一个新目录：

```bash
python3 tools/freeze.py ../runs/step_v2_example
```

训练/评估命令只在容器中执行，并通过`tools/run_guard.py`核查退出、产物和物理错误。以下是固定副本内的工程命令参数，**不是已启动的正式训练**：

```text
-m step_demo train --training-config configs/engineering.json --run-dir ../engineering \
  --asset-dir ../asset --model-dir ../../../robot_description --stop-after 4

-m step_demo train --run-dir ../engineering --asset-dir ../asset \
  --model-dir ../../../robot_description --resume

-m step_demo evaluate --run-dir ../engineering --asset-dir ../asset \
  --model-dir ../../../robot_description --output ../evaluation --eval-envs 16
```

训练保留origin、随机初始化指纹、完整优化器和更新计数。恢复严格核对同一次运行、配置、源码和资产；仿真回合及GRU回合记忆重新开始。评估记录完整2ms物理采样到`physics.jsonl.gz`，使用`python3 -m step_demo.review RUN EVALUATION --output analysis.json`独立复核。

工程短训不作为策略成功率、模型交付或sim2sim证明；其权重和轨迹不得迁移到正式训练。运行状态以批次状态文件及实际进程为准。
