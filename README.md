# PetBot Step RL Demo

从随机初始化训练直线上低台阶的独立策略：前轮提前抬起、越过前沿后落到平台顶面，后轮保持支撑并滚上平台；随后恢复默认髋角，沿原直线前进一小段并停稳。

**当前状态：已实现 Isaac Lab/PhysX 环境、RSL-RL/PPO 训练、同次恢复和重载评估。工程验收正在进行，尚未证明策略学会任务；不包含 sim2sim。**

## 固定首版场景

- 默认姿态、静止开始，平台前沿位于初始整机质心沿前进方向的0.30m处；平台高0.02m，边沿与行驶方向成90°，有足够宽度和长度。
- 速度由策略决定，允许减速、短暂停顿，不主动倒车。前轮不得碰撞立面或棱边，允许落到顶面；后轮允许接触台阶并滚上。
- 两个动作量为共同髋角和共同后轮速度，左右目标严格相同；头部/脖子和耳朵保持默认位置，前轮被动。对称目标不等于物理轨迹必然对称，仍须验收原直线上的实际运动。
- 台阶高度、初始距离和边沿角度每回合只输入一次；之后由自身传感信息估计运动。不得每步向策略提供真实剩余距离、真实位姿或接触判定。
- 不依赖其他demo、历史检查点、冻结父网络、教师或示范数据。源码经验可以独立审查后复用；新训练必须随机初始化。

完整契约见[任务规格](docs/task.md)，实现状态和下一步见[架构与实施顺序](docs/architecture.md)。

## 当前可运行的检查

在本项目目录使用Python 3.10或更高版本，无第三方依赖，也不启动Isaac/训练：

```bash
python3 -m step_demo check
python3 -m unittest discover -s tests -v
```

源模型默认相对本项目读取`../../robot_description`。独立检出时可显式指定模型目录：

```bash
python3 -m step_demo check --model-dir /path/to/robot_description
```

`check`输出模型文件指纹、物理限位、场景和接口维数。模型和物理参数只读，地形将由step自己的场景适配层创建。`train`和`evaluate`只允许在专用容器执行。训练流程见下方。

## 目录

| 路径 | 职责 |
|---|---|
| `configs/step_v1.json` | 首版场景、接口与待验证的验收初值 |
| `step_demo/model.py` | 只读解析MJCF执行器限位、计算模型文件指纹 |
| `step_demo/control.py` | 两维归一化动作到左右同步目标的映射 |
| `step_demo/observations.py` | 一次性地形锁存、自身状态/运动估计与actor输入 |
| `step_demo/interfaces.py` | 标量接口参考；具体后端为 `isaac_env.py` |
| `step_demo/evaluation.py` | 独立验收流程；用于离线复核的标量完成判据 |
| `tests/` | 合成输入验证动作、观测边界及结束判据；非仿真证据 |
| `docs/` | 用户契约、实现边界与下一阶段门槛 |

本项目由`repo`物化，源码在`demoproj/step_rl_demo`分支。训练只在专用容器内使用固定提交副本，产物放主仓库忽略的`demo_project/runs/`；不启动周期定时器或自动长训队列。

## 容器与训练

运行时：Isaac Lab 3.0.0-beta2、PyTorch 2.10.0+cu128、RSL-RL 5.4.1、Python 3.12。专用容器为 `petbot-step-train`。`docker/Dockerfile`记录构建依赖；`tools/container.py`创建容器并将源模型只读挂载，默认选择物理GPU 2。已存在容器不要重复创建。

先在容器构建本demo派生资产：

```bash
docker exec petbot-step-train /isaac-sim/python.sh tools/prepare_asset.py
```

完成代码提交后，在宿主机冻结源码与派生资产。每次使用新目录：

```bash
python3 tools/freeze.py ../runs/step_run_example
```

然后使用宿主机监督入口，在固定副本内启动PPO：

```bash
python3 tools/run_guard.py --workdir /workspace/runs/step_run_example/source \
  --log ../runs/step_run_example/train.log --status ../runs/step_run_example/train/status.json -- \
  -m step_demo train --run-dir ../train --asset-dir ../asset --model-dir ../../../robot_description
```

默认1024环境、1000次更新。工程检查使用 `--training-config configs/engineering.json`（128环境、6更新）；加 `--stop-after 4`验证在更新边界中断，再以相同入口和 `--resume`完成剩余预算。恢复时不改配置，不指定外来检查点；若进程意外退出，恢复最近一次完整保存的更新，当前未保存rollout丢弃。日志使用新名字。

独立进程重载评估，把上述监督参数的日志/状态改为新路径，执行：

```text
-m step_demo evaluate --run-dir ../train --asset-dir ../asset --model-dir ../../../robot_description --output ../evaluation --eval-envs 32
```

查看 `origin.json`、`initialization.json`、`latest.json`、`status.json`、`*.guard.json`和评估`result.json`；轨迹在`trace*.jsonl`，逐回合终局在`episodes.jsonl`。监督器检查真实产物和物理/Python错误，不能只看Isaac进程退出码。固定首版没有随机地形，多个并行环境不能冒充独立泛化测试。
