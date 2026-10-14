# recovery_agile_01 — PetBot 首个恢复 demo

基于 WBC-AGILE `6830cf995714e81c91ce63247e8e016d36e28f14`；任务 `PetBot-Recovery-AGILE-v0`。
复用 AGILE 的任务注册、pre-learn 生命周期、RSL-RL 5 适配与训练/检查点/TensorBoard 入口；借鉴 robot_lab 的位置/速度 action 分组。未继承 T1 的镜像、提升辅助、非足端触地惩罚或高度目标。

## 当前验证边界

2026-10-13 在训练服务器 `petbot-train` 容器内验证：**USD 转换、PhysX 导入、任务实例化、pre-learn（默认站姿探针 + 倒地初态采集）与冒烟训练全部跑通**；2026-10-13/14 又完成首轮 11 组（2048 envs × 3000 iteration）扫描与确定性独立评估。**已实测的结论是训练效果尚未出现**（最好 5 组仅与开环"保持默认构型"地板基线持平），详见[首版扫描记录](PETBOT_SWEEP_01.md)。仍无策略导出与 MuJoCo 策略推理验证。

### 服务器验证记录（2026-10-13）

| 步骤 | 结果 |
|---|---|
| `prepare --headless` | 通过。派生资产 `artifacts/petbot_asset/<identity>/`：USD `usd/robot_description/robot_description.usda`、11 个刚体、质量 **1.65254 kg**（与 MJCF 一致）、9 个关节名等于契约名 |
| 默认站姿探针 | 通过（≥90% 环境连续稳定 0.5 s），实测默认站姿高度 **0.08427 m** |
| 倒地初态采集 | 64 envs：256/256（5 批）；1024 envs：16384/16384（20 批，接受率约 80%），缓存于 `artifacts/fallen_states/` |
| 冒烟训练 64 envs × 5 iter | 通过：约 755 steps/s、2.7 s/iteration，写出 `model_0/model_4.pt` 与 TensorBoard 事件，`episode` 结束前 `Recovery/*` 与奖励项均正常记录 |
| 首版规模 1024 envs × 5 iter | 通过：约 8000 steps/s、4.0 s/iteration，单卡（RTX 4090）显存峰值约 3.7 GB；据此推算 10000 iteration ≈ 11 h，另有一次性采集约 6 min。**首版口径已按用户决定改为 2048 envs**（见下） |

## 首版训练与扫描（2026-10-13/14）

- 口径（用户决定）：**2048 envs × 3000 iteration**（1024 太少），11 组变体各跑**固定 commit 的独立副本** `demo_project/runs/<变体>/`，先宽扫积累经验；9 组并行约 10 h 跑满，调度与探活脚本为工作区 `demo_project/demo_env/{launch_variants.sh,runs_probe.sh}`。
- 结果：**没有任何一组学到"从倒地起身"**。最好 5 组确定性评估 0.225~0.227，而"恒定把 5 个关节指令到默认角、后轮不转"的开环地板基线就有 **0.213**（恒定零动作 0.000）；其余 6 组 0.000~0.084，且大量 episode 因越界/非法状态提前结束。
- 三条机制（均为实测）：① 训练期 `Recovery/success_rate` 测的是带探索噪声的策略，不能用于组间比较，比较必须用确定性独立评估；② 约 21% 初态本就满足成功判据 → 成功率与 `stable` 奖励可被"不动"拿到，需加"必须发生恢复"的门控；③ 伺服增益（0.5×/1×/2×）不是瓶颈，控制率、回合长度、加重静止惩罚、更大探索噪声都更差。
- 设计思路、执行现象、实现坑与复现清单见 **[首版扫描记录 PETBOT_SWEEP_01.md](PETBOT_SWEEP_01.md)**。

两条与实验设计强相关的边界：

- **CLI 覆盖只对 runner（agent）侧有效**（`agent.*`）；env 字段（`rewards.*`、`decimation`、`episode_length_s`）Hydra 不接受，且这类字段在 `__post_init__` 里赋值、覆盖顺序不可靠 → 这类变体一律做**代码副本**并在副本内 commit（保持工作区干净，`pre_learn` 会拒绝脏树）。
- 日志必须写在副本**之外**（`demo_project/runs/_logs/<变体>.log`）：副本根目录出现未跟踪文件同样会被脏树检查拒绝。

评估口径：确定性策略 + `split=eval` 初态库 + 独立 seed + 512 episodes，按初态分类（直立 / 倒置 / 侧躺）拆分，并**始终与两个地板基线对比**（恒定零动作、恒定保持默认构型）；总成功率含"落地后已近站立"的样本，只看总数会被"不动"骗过。

两处服务器适配（已提交本分支）：

1. **容器内 git 可用性**：本 demo 的 `.git` 是 repo 的**相对**符号链接（`../../.repo/projects/demo_project/...`），bare repo 的 `objects/hooks/rr-cache` 又指向 `../../../project-objects/petbot2_rl.git/...`，而 `pre_learn` 开头的脏树检查会调用 git。容器内必须复刻宿主相对层级（demo 挂到 `/workspace/demo_project/recovery_agile_01`，并挂 `/workspace/.repo/projects/demo_project`、`/workspace/.repo/project-objects`），否则训练在 pre_learn 处直接失败。
2. **导入关节名**：Lab 3 的 MJCF 导入器在关节与所在 body 下**同名的 geom prim** 重名时给关节加 `_1` 后缀（本模型 9 个关节全部如此），任务会在 `init_state.joint_pos` 的正则匹配处报错。`adapt_usd` 现在只改 USD 的 prim 命名：把同名 geom 让到 `<name>_geom`，再把关节改回契约名，并校验非 fixed 关节名集合等于 `JOINT_ORDER`；几何、碰撞与物理参数不变（派生 MJCF 的 `worldbody` 与源保持逐字节一致）。

## 目录与版本

- `agile/petbot/`：PetBot 模型契约、派生资产工具、任务配置、初态库、奖励与指标。
- `agile/rl_env/tasks/stand_up/petbot/`：注册。
- `petbot_tests/`：无 Isaac Sim 的 CPU 测试。
- `artifacts/`：生成的 MJCF/USD 和初态库；`logs/rsl_rl/recovery_agile_01/`：检查点、TensorBoard 和运行契约，均不提交 Git。
- 原始 MJCF/STL 从工作区根的 `robot_description/` **只读**加载。可以整体挂载工作区；单独挂载源模型时设置 `PETBOT_MODEL_DIR`。代码无宿主绝对路径。
- demo 分支 `demoproj/recovery_agile_01`，由主仓库 `default.xml` 检出到 `demo_project/recovery_agile_01`。修改前同步，完整修改后提交推送；同一 demo 单端修改，在跑实验使用固定 commit 的独立副本。禁止对运行副本直接 `repo sync`。训练入口在采集前拒绝有未提交改动的 demo。

## 容器内使用

使用与上游 pin 匹配的 Python 3.12 / Isaac Lab 3.0.0b2 / Isaac Sim 6 / RSL-RL 5.4.1 环境；安装见原 README、pyproject.toml、uv.lock 和官方 workflows/Dockerfile。必须让容器导入**当前 demo**的 agile 包，而不是 reference_project 下的另一份；可在当前目录 `python -m pip install --no-deps -e .`，并运行上游 `python scripts/verify_rsl_rl.py` 核查补丁。首版不额外修改依赖锁。训练容器与挂载约定见工作区 `demo_project/demo_env/env.sh`（未纳入版本管理）；容器以 root 运行（3.0.0-beta2 镜像默认 uid 1000，否则写不了宿主属主的 checkout），因此生成的 `artifacts/`、`logs/` 在宿主侧属于 root，从宿主删除需经由容器执行。

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

# 正式首版口径（2026-10-13 用户决定：2048 envs）。1024 envs 实测 4.0 s/iteration、显存 3.7 GB。
python scripts/train.py --task PetBot-Recovery-AGILE-v0 --headless \
  --num_envs 2048 --max_iterations 10000 --seed 42 --logger tensorboard

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

派生 MJCF 只更换五个 motor 为原生 position servo，保留源力矩上限、几何/质量/惯量/关节参数和接触规则。USD 使用凸分解碰撞体，导入后检查总质量和轮名称，明确过滤轮子与机器人自身的碰撞；导入器给同名关节加的 `_1` 后缀在 `adapt_usd` 里通过改名同名 geom prim 消除（见「服务器验证记录」）。PhysX 不完全等价于 MuJoCo：用 minimum 材料组合匹配各部位对地滑动摩擦（机体/前轮 0.1、后轮 1.4），自接触摩擦、扭转/滚动摩擦和凸分解碰撞几何需要后续检验；这是转换限制，不是源模型调参。若模型或转换代码改变，必须重新生成资产和初态库。

## 本地检查

```bash
python -m pytest -q --confcutdir=petbot_tests petbot_tests
ruff check agile/petbot agile/rl_env/tasks/stand_up/petbot petbot_tests scripts/petbot_eval.py
```

依赖 `torch`（CPU 即可）、`numpy`、`mujoco`、`pytest`，不加载 Isaac Sim。MuJoCo 测试只编译并比较模型数组，不启动训练或策略运行。

## 接口核查依据

- Isaac Lab API 源码核对版本：`v3.0.0-beta2`，commit `28a37cecdd433c22d9eabd6a5954add9f13a8951`；包括 MJCF converter、implicit actuator、joint action、manager reset 顺序与 ProxyArray 的 `.torch` 接口。
- [官方 Lab 3 迁移说明](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/migration/migrating_to_isaaclab_3-0.html)。具体配置以该 tag 的源码为准。
- 初始配置与性能已在服务器实测（见[首版扫描记录](PETBOT_SWEEP_01.md)）；本地检查记录不替代部署结果。
