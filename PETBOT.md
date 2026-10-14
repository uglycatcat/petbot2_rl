# recovery_agile_01 — PetBot 首个恢复 demo

基于 WBC-AGILE `6830cf995714e81c91ce63247e8e016d36e28f14`；任务 `PetBot-Recovery-AGILE-v0`。
复用 AGILE 的任务注册、pre-learn 生命周期、RSL-RL 5 适配与训练/检查点/TensorBoard 入口；借鉴 robot_lab 的位置/速度 action 分组。未继承 T1 的镜像、提升辅助、非足端触地惩罚或高度目标。

## 当前验证边界

2026-10-13 在训练服务器 `petbot-train` 容器内验证：**USD 转换、PhysX 导入、任务实例化、pre-learn（默认站姿探针 + 倒地初态采集）与冒烟训练全部跑通**。仍未验证的是训练效果本身——没有跑满 iteration、没有独立评估、没有策略导出或 MuJoCo 推理，因此不能宣称策略有效。

### 服务器验证记录（2026-10-13）

| 步骤 | 结果 |
|---|---|
| `prepare --headless` | 通过。派生资产 `artifacts/petbot_asset/<identity>/`：USD `usd/robot_description/robot_description.usda`、11 个刚体、质量 **1.65254 kg**（与 MJCF 一致）、9 个关节名等于契约名 |
| 默认站姿探针 | 通过（≥90% 环境连续稳定 0.5 s），实测默认站姿高度 **0.08427 m** |
| 倒地初态采集 | 64 envs：256/256（5 批）；1024 envs：16384/16384（20 批，接受率约 80%），缓存于 `artifacts/fallen_states/` |
| 冒烟训练 64 envs × 5 iter | 通过：约 755 steps/s、2.7 s/iteration，写出 `model_0/model_4.pt` 与 TensorBoard 事件，`episode` 结束前 `Recovery/*` 与奖励项均正常记录 |
| 首版规模 1024 envs × 5 iter | 通过：约 8000 steps/s、4.0 s/iteration，单卡（RTX 4090）显存峰值约 3.7 GB；据此推算 10000 iteration ≈ 11 h，另有一次性采集约 6 min。**首版口径已按用户决定改为 2048 envs**（见下） |

## 首版训练方案（2026-10-13 用户决定，已部署）

口径：**2048 envs × 3000 iteration**，先宽扫积累经验，跑满后按曲线决定续训组；每组跑在**固定 commit 的独立副本** `demo_project/runs/<变体>/`（`git clone` 自本地裸库，禁止对运行副本 `repo sync`）。日志写 `demo_project/runs/_logs/<变体>.log`——**不能写在副本里**，副本根目录出现未跟踪文件会让 `pre_learn` 的脏树检查拒绝启动；检查点与 TensorBoard 在副本自己的 `logs/rsl_rl/recovery_agile_01/`。

调度：`demo_project/demo_env/launch_variants.sh {start|status|log <v>|stop}`（`runs_probe.sh` 统计容器内在跑的组与卡位）。按卡分配：**GPU3 上限 6 组、GPU2 上限 3 组**（GPU2 上还有 AGILE 标准训练，必须留显存），超出上限的自动排队补位；实测 9 组并行时 2048 envs 约 7~8 s/iteration。

| 变体 | 卡 | 相对默认的唯一改动 |
|---|---|---|
| `base_s42` / `base_s43` | 3 / 3 | 无 / 仅 seed 43（量化种子方差） |
| `gains2x_s42` / `gains_down_s42` | 3 / 3 | kp/kd ×2 / ×0.5 → 构成 [0.5×, 1×, 2×] 增益扫描 |
| `decimation5_s42` | 3 | 控制 100 Hz（默认 10 分频 = 50 Hz） |
| `reward_still_s42` | 2 | `still` 0.5→2.0、`action_rate` −0.01→−0.02、`effort` −0.002→−0.005 |
| `reward_config_heavy_s42` | 3 | `configuration` 3.0→6.0、`upright` 2.0→1.0 |
| `episode15_s42` | 2 | episode 10 s → 15 s |
| `spawn1m_s42`（排队） | 3 | 落地高度 0.6 → 1.0 m |
| `ppo64_s42` | 2 | `num_steps_per_env` 32→64、`learning_rate` 3e-4→2e-4 |
| `ppo_explore_s42`（排队） | 3 | `entropy_coef` 0.005→0.02、`init_noise_std` 0.7→1.0 |

**CLI 覆盖的边界（实测）**：Hydra 只接受**runner（agent）配置**的覆盖（`agent.*`，如 `agent.petbot_collection.spawn_height=1.0`）；**env 配置字段不被接受**——`rewards.still.weight=2.0` 会报 `Could not override 'rewards.still.weight' … Key 'rewards' is not in struct`。因此奖励、`decimation`、`episode_length_s` 这类 env 字段一律做**代码副本**（改完在副本内 commit 保持干净），只有 actor-critic 侧参数用 `agent.*` 覆盖。

复现要点：`prepare`（资产转换）**串行**执行，之后把已备好的资产与初态库拷进各副本——资产身份只含源模型内容、适配器版本与位置增益，不含初态库与奖励，所以只有改了增益或模型、或改了初态/控制率的变体才需要自己重新采集初态库（每组约 5 min）。

并发注意：多个 Kit 实例共用 `cache/kit` 会打 key-value 锁警告（无害）；不要并发跑 `prepare`。

评估：跑满后对每组用 `scripts/petbot_eval.py`（独立 seed、`split=eval` 初态库）打分，报 `Recovery/success_rate`、`stable_at_end`、`time_to_success_s`，并**按初态分类**（倒置 / 侧躺 / 已近站立）拆开看——总成功率含"落地后已近站立"的样本，不能只看总数；只有稳定的组才进入策略导出与 MuJoCo sim2sim。

## 首版并行扫描结果（2026-10-14）

11 组各跑满 2048 envs × 3000 iteration（约 6.6 亿步，8~9 组并行约 10 h），随后对每组用**确定性策略**做独立评估（`scripts/petbot_eval.py`，`split=eval` 初态库、512 episodes、seed 20261014）：

| 变体 | 确定性评估成功率 | 侧躺类 | 中位成功用时 | 末态稳定 |
|---|---|---|---|---|
| `base_s42` / `gains2x_s42` / `gains_down_s42` / `ppo64_s42` | 0.227 | 0.249 | 1.42–1.48 s | 0.21–0.23 |
| `base_s43` | 0.225 | 0.247 | 1.48 s | 0.213 |
| `reward_config_heavy_s42` | 0.084 | 0.094 | 3.10 s | 0.018 |
| `spawn1m_s42` | 0.053 | 0.059 | 2.62 s | 0.006 |
| `reward_still_s42` / `episode15_s42` / `decimation5_s42` / `ppo_explore_s42` | 0.000 | 0.000 | — | 0.000 |

两组基线（**同一批 512 初态**，用来判断上面的数字意味着什么）：

| 基线 | 成功率 |
|---|---|
| 恒定零动作（`demo_env/zero_action_eval.py`，动作全 0） | 0.000 |
| **恒定保持默认构型**（关节指令默认角、后轮不转，开环，动作 `-0,-0.5,-0.5,0,0,0,0`） | **0.213**（侧躺类 0.234） |

**结论：最好的 5 组（0.225~0.227）与"什么都不学、只把 5 个关节摆回默认角再不动"的开环基线（0.213）没有实质差别**；其余 6 组低于该基线（0~0.084，其中 4 组为 0，且相当多 episode 因越界 / 非法状态提前结束，说明策略在乱动）。即 3000 iteration 内**没有任何一组学到"站起来"**——评估成功率主要来自初态库中本来就站得住的约 21%~23% 状态。

三点实测机制（对下一轮设计有直接约束）：

1. **训练期 `Recovery/success_rate` 不能用来比较变体**：训练动作是采样的（`init_noise_std=0.7` 的探索噪声），"所有条件连续 1 s"在噪声下几乎不可达，11 组里只有 `base_s43` 出现过非零（0.175），其余恒 0；而同一检查点在**确定性**评估下是 0.225~0.227。比较必须用独立评估，报告里要区分"探索策略"与"确定性策略"。
2. **约 21% 的初态本来就满足成功条件**（零动作不行，但"保持默认构型不动"就行）→ 当前的 `Recovery/success_rate` 与 `stable` 奖励（权重 5.0，最大正项）都可以被"不动"拿到。这与 PetBot 的目标冲突（要的是"从倒地起身"，不是"站住"），需要给指标与奖励加"必须发生恢复"的门控：例如按初态分类统计、排除初态即满足条件 / 近直立的样本，或只在"初始未满足条件"的样本上计分。
3. **伺服增益不是当前瓶颈**：kp/kd ×0.5、×2 与默认落在同一水平（0.227）；控制率 100 Hz、回合 15 s、加大静止/动作率惩罚这几种改法反而更差（0~0.084）。另外 `ppo_explore`（更大探索噪声）为 0，与第 1 点的噪声结论一致。

评估脚本的两个坑（一个已修、一个已知无碍）：`restore_cfg` 会因类型不匹配丢弃 `seed` 与整个 `scene` 段（评估跑在同一代码副本里，`scene` 默认值一致，故当前无影响）；`petbot_eval.py` 的入口块 `try: main() finally: app.close()` 会在异常时吞掉 traceback 并静默退出，排查时用 `demo_env/run_eval_debug.py` 包装运行。

下一轮建议：训练预算提到 10000+，并同时修第 1、2 两点信号问题（探索噪声衰减课程 + 成功/奖励门控只在"初态未满足条件"的样本上给分），再用同一规模扫描一次。

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
- 初始配置与性能均待服务器验证；本地检查记录不替代部署结果。
