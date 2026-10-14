# PetBot 机器人模型

本目录保存自恢复任务使用的当前机器人模型，是**模型参数的唯一来源**：MJCF、STL 及其中的物理参数**禁止修改**（见 [AGENTS.md](../AGENTS.md) 的全局规则）；本 README 中**对机器人的文字描述**可以更新。

## 文件

| 路径 | 用途 |
|---|---|
| [mjcf/scene.xml](mjcf/scene.xml) | MuJoCo 场景入口：地面、接触对、光照，并 include 机器人本体 |
| [mjcf/robot_description.xml](mjcf/robot_description.xml) | 本体：几何、质量惯量、关节、执行器与传感器 |
| `meshes/` | 10 个 STL 网格（毫米单位，缩放与坐标变换由 MJCF 指定） |

（本机曾有一个预留的空 `urdf/` 目录；git 不跟踪空目录，clone 后不会出现，也不是本项目需要的资产。）

## 机器人概况

以下数值全部来自当前 MJCF；任何描述与文件冲突时**以文件为准**：

| 项 | 值 |
|---|---|
| 整机质量 | 1.65254 kg（下身 0.30、头 0.43、左右车架各 0.36，其余为耳 / 轮 / 浮动基座） |
| 自由度 | 浮动基座 + 9 个旋转关节；MuJoCo 加载结果 `nq=16, nv=15, nu=7` |
| 受控通道 | 5 个位置（`head`、`left_ear`、`right_ear`、`left_frame`、`right_frame`）+ 2 个后轮速度；左右前轮被动 |
| 执行器 | 五个有界关节用 `motor`（力矩），两个后轮用 `velocity` |
| 时间 | `timestep 0.002 s`，重力 `0 0 -9.81` |
| 接触 | 自碰开启；轮子只与地面碰撞（不与机体 / 其他轮碰撞）；前轮 geom 显式设滑动摩擦 `0.1` 且 `priority="1"` |
| 摩擦 | 关节 `frictionloss` 与 geom `friction` 都是**经验值、无实测** |

## 后续 RL 接口

已确定为 **5 路位置目标（rad）+ 2 路后轮速度目标（rad/s）**，由仿真内部伺服执行；该转换尚未实施。见 [当前需求基线](../docs/current_scope.md)。

## 边界

- 早期键盘驾驶在 `robot_controller_demo/driver_demo/`，曾用外部位置 PID 驱动那五个力矩执行器；它当年把前轮滑动摩擦调到 `0.1`，属历史模型改动，不要再据此改动模型。同目录的固定姿态恢复见 [fixed_pose_recovery.md](../docs/fixed_pose_recovery.md)，同样不改本目录的模型。
- 本模型与另一个 ROS 工作区里用于 ToF/ROS 演示的机器人描述用途不同。
- 历史报告（`docs/reports/`）中的参数与快照只作追溯，不随当前模型更新。
