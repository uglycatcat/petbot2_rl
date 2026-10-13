# recovery_agile_01 — PetBot 首个恢复 demo

基于 WBC-AGILE `6830cf995714e81c91ce63247e8e016d36e28f14`；任务 `PetBot-Recovery-AGILE-v0`。
复用 AGILE 的任务注册、pre-learn 生命周期、RSL-RL 5 适配与训练/检查点/TensorBoard 入口；借鉴 robot_lab 的位置/速度 action 分组。未继承 T1 的镜像、提升辅助、非足端触地惩罚或高度目标。

## 当前验证边界

本阶段只在桌面编写与检查代码，不连接服务器、不启动训练。CPU 测试覆盖源模型/派生模型物理参数一致性、动作配置、SO(3) 采样、成功保持、采集伺服关闭恢复及重置日志。**USD 转换、PhysX 导入与接触、任务实例化和训练尚需服务器容器验证**；不能据此宣称训练已跑通或策略已有效。

## 目录与版本

- `agile/petbot/`：PetBot 模型契约、派生资产工具、任务配置、初态库、奖励与指标。
- `agile/rl_env/tasks/stand_up/petbot/`：注册。
- `petbot_tests/`：无 Isaac Sim 的 CPU 测试。
- `artifacts/`：生成的 MJCF/USD 和初态库；`logs/rsl_rl/recovery_agile_01/`：检查点、TensorBoard 和运行契约，均不提交 Git。
- 原始 MJCF/STL 从工作区根的 `robot_description/` **只读**加载。可以整体挂载工作区；单独挂载源模型时设置 `PETBOT_MODEL_DIR`。代码无宿主绝对路径。
- demo 分支 `demoproj/recovery_agile_01`，由主仓库 `default.xml` 检出到 `demo_project/recovery_agile_01`。修改前同步，完整修改后提交推送；同一 demo 单端修改，在跑实验使用固定 commit 的独立副本。禁止对运行副本直接 `repo sync`。训练入口在采集前拒绝有未提交改动的 demo。

## 容器内使用（本次不执行）

使用与上游 pin 匹配的 Python 3.12 / Isaac Lab 3.0.0b2 / Isaac Sim 6 / RSL-RL 5.4.1 环境；安装见原 README、pyproject.toml、uv.lock 和官方 workflows/Dockerfile。必须让容器导入**当前 demo**的 agile 包，而不是 reference_project 下的另一份；可在当前目录 `python -m pip install --no-deps -e .`，并运行上游 `python scripts/verify_rsl_rl.py` 核查补丁。首版不额外修改依赖锁。

在 demo 根目录运行：

```bash
# 仅源模型适配，不启动 Isaac；仅生成派生副本。
python -m agile.petbot.prepare --mjcf-only

# 容器内准备 USD；失败时不写入可用 asset.json 标记。
python -m agile.petbot.prepare --headless

# 先检查小规模完整训练链路。初态库数量需相应缩小。
python scripts/train.py --task PetBot-Recovery-AGILE-v0 --headless \
  --num_envs 64 --max_iterations 5 --logger tensorboard \
  agent.petbot_collection.num_states=256

# 正式首版初值，尚未验证吞吐和显存。
python scripts/train.py --task PetBot-Recovery-AGILE-v0 --headless \
  --num_envs 1024 --max_iterations 10000 --seed 42 --logger tensorboard

# RSL-RL 续训，iterations 是本次继续学习的迭代数。
python scripts/train.py --task PetBot-Recovery-AGILE-v0 --headless \
  --resume True --checkpoint logs/rsl_rl/recovery_agile_01/<run>/model_100.pt \
  --max_iterations 1000 --logger tensorboard

tensorboard --logdir logs/rsl_rl/recovery_agile_01

# 后续独立评估，使用不同于训练库的采集种子。
python scripts/petbot_eval.py --headless --episodes 512 \
  --checkpoint logs/rsl_rl/recovery_agile_01/<run>/model_100.pt
```

若需单独生成评估初态，使用不同 seed 并设置 `agent.petbot_collection.split=eval`。训练时不混入评估库；独立评估入口见 `scripts/petbot_eval.py`。上游的人形评估判据不适用于 PetBot。

## 控制与观测契约

固定动作顺序：`head, left_ear, right_ear, left_frame, right_frame, left_rear_wheel, right_rear_wheel`。
前五维 [-1,1] 线性映射到源模型关节全范围（不是默认角的小偏移）；后两维映射到 [-25,25] rad/s。绝对位置目标再按源限位裁剪，轮速按源控制范围裁剪。**零动作对应位置区间中点，并不等于默认零关节角**。

头/耳/轮架力矩上限分别 0.72 / 0.10 / 1.20 N·m；后轮 0.38 N·m。五位置 kp 初值 3/1.5/1.5/4/4，kd 0.10/0.035/0.035/0.12/0.12；后轮 kv 0.14。源关节摩擦、armature 和被动阻尼单独映射；前轮零驱动。所有增益均待响应试验确认。

actor 为 27 维：本体角速度×0.2、投影重力、五关节相对默认角、9 关节速度×0.05、上一动作。速度顺序为五位置、两后轮、两前轮。轮绝对角不进观测。critic 为 35 维，另加本体线速度、离地高度、四轮竖直接触力×0.1。四元数 wxyz；原物理步长 0.002 s，decimation 10（策略 50 Hz），episode 10 s。首版无观测噪声、域随机化、提升辅助或对称增强。

## 初态与目标

先在位置伺服下测量默认构型沉降高度与稳定支撑；达不到门槛则报错，优先检查资产与控制。初态按 SO(3) 均匀根朝向、五关节合法全范围、连续轮角 [-pi,pi] 随机，初始速度为零，从 0.6 m 落地。采集时关闭驱动刚度/阻尼，保留被动摩擦/阻尼，结束后必定恢复驱动。

默认采集 16384 个状态，每批沉降 3 s，末段 0.5 s 连续低速且有地面支撑才接受；保留实际残余速度，不把运动状态强行清零。采集不足报错，不静默退回站姿训练。沉降后的分布不再保证方向均匀，仍可能含近站立初态。初态库缓存绑定源模型内容、USD、关节顺序、采集种子/参数、物理步长和实现版本。

目标为默认姿态和五关节构型、四轮支撑、低身体/关节/轮速保持 1 s；初值容差：倾角 15°、单关节偏差 0.15 rad、高度偏差 0.02 m、线速度 0.05 m/s、角速度 0.15 rad/s、位置关节速度 0.15 rad/s、轮速 0.5 rad/s。水平位置/航向自由，但出环境单元半径 1 m 结束，避免跨环境污染。成功后继续运行至 episode 结束，用于学习保持。

奖励是正向直立、接近默认构型/高度、近直立时运动惩罚、稳定保持，以及温和动作变化和力矩惩罚；高度奖励是接近测量目标，非越高越好。记录 `Recovery/success_rate`（本 episode 曾连续成功）、`Recovery/stable_at_end`（末步仍满足保持）、`Recovery/time_to_success_s`（仅成功样本），以及各条件末态通过率。成功率包含随机落地后已经接近站立的样本，不能当作严格倒置恢复成功率。

## 模型转换边界

派生 MJCF 只更换五个 motor 为原生 position servo，保留源力矩上限、几何/质量/惯量/关节参数和接触规则。USD 使用凸分解碰撞体，导入后检查总质量和轮名称，明确过滤轮子与机器人自身的碰撞。PhysX 不完全等价于 MuJoCo：用 minimum 材料组合匹配各部位对地滑动摩擦（机体/前轮 0.1、后轮 1.4），自接触摩擦、扭转/滚动摩擦和凸分解碰撞几何需要后续检验；这是转换限制，不是源模型调参。若模型或转换代码改变，必须重新生成资产和初态库。

## 本地检查

```bash
python -m pytest -q --confcutdir=petbot_tests petbot_tests
ruff check agile/petbot agile/rl_env/tasks/stand_up/petbot petbot_tests scripts/petbot_eval.py
```

依赖 `torch`（CPU 即可）、`numpy`、`mujoco`、`pytest`，不加载 Isaac Sim。MuJoCo 测试只编译并比较模型数组，不启动训练或策略运行。

## 接口核查依据

- Isaac Lab API 源码核对版本：`v3.0.0-beta2`，commit `28a37cecdd433c22d9eabd6a5954add9f13a8951`；包括 MJCF converter、implicit actuator、joint action、manager reset 顺序与 ProxyArray 的 `.torch` 接口。
- [官方 Lab 3 迁移说明](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/migration/migrating_to_isaaclab_3-0.html)。具体配置以该 tag 的源码为准。
- 初始配置与性能均待服务器验证；本地检查记录不替代部署结果。
