# PetBot 首版并行扫描（2026-10-13/14）：设计思路、执行现象与结论

本文件是 `recovery_agile_01` 首轮参数扫描（11 组，2048 envs × 3000 iteration）的完整记录：**为什么这样设计实验、实际执行中出现了什么现象、结论对下一轮的约束**。控制契约、容器与评估入口见 [PETBOT.md](PETBOT.md)。

## 0. 一句话结论

11 组跑满（约 6.6 亿步）后，**没有任何一组学到"从倒地起身"**：最好的 5 组确定性评估成功率 22.5%~22.7%，而"什么都不学、只把 5 个关节指令到默认角、后轮不转"的开环基线就有 **21.3%**；其余 6 组低于该基线。瓶颈是**信号**（成功判据与权重最大的正奖励可以被"不动"拿到），不是训练预算或超参。

## 1. 设计思路

### 1.1 出发点

- PetBot 的默认站姿是四轮着地的静稳定姿态，**"站住"不需要学**；要学的是"倒地后起身并回到默认构型"。因此判据/奖励必须围绕"是否发生了恢复"，而不是"最终是否直立"。
- demo 的多数参数是工程初值（伺服增益、奖励权重、控制率、初态与容差都是估的），所以首轮目标是**建立可比较的评估口径与"地板基线"**，而不是直接冲成功率。

### 1.2 口径与预算（用户决定）

- **2048 envs**（1024 太少被否）× **3000 iteration**：宽扫优先，跑满后按曲线决定哪些组续训。
- 每组跑在**固定 commit 的独立副本** `demo_project/runs/<变体>/`（`git clone` 自本地裸库），禁止对运行副本 `repo sync`。

### 1.3 扫描轴与选择理由

每组只改一个因素，便于归因；共 11 组。

| 变体 | 轴 | 为什么选它 |
|---|---|---|
| `base_s42` / `base_s43` | 基线 / 种子 | 参考项目教训：同配置可差数倍。没有重复种子，无法判断其它组的差异是不是噪声 |
| `gains2x_s42` / `gains_down_s42` | 伺服增益 kp/kd ×2 / ×0.5 | 文档里唯一自认"未实测"的工程初值；构成 [0.5×, 1×, 2×] 扫描 |
| `decimation5_s42` | 控制率 50 → 100 Hz | 位置伺服带宽与动作分辨率是否限制恢复 |
| `episode15_s42` | 回合 10 → 15 s | 起身可能需要更长时间，10 s 是否截断了成功 |
| `reward_still_s42` | `still`/`action_rate`/`effort` 加重 | 针对"站住了但还在晃"这一失败模式 |
| `reward_config_heavy_s42` | `configuration` 加重、`upright` 减轻 | PetBot 的目标是回到**默认构型**，不只是直立 |
| `spawn1m_s42` | 初态落地高度 0.6 → 1.0 m | 更剧烈的落地初态，检验恢复范围 |
| `ppo64_s42` | `num_steps_per_env` 32→64、lr 3e-4→2e-4 | 采样效率与收敛速度 |
| `ppo_explore_s42` | `entropy_coef` 0.005→0.02、`init_noise_std` 0.7→1.0 | 探索强度是否是瓶颈（也是第 2.4 节第 1 条的对照） |

### 1.4 资源与调度

- 卡位：0/1 号是他人的负载、2 号跑 AGILE 标准训练 → 只用 2/3。**GPU3 上限 6 组、GPU2 上限 3 组**，给 AGILE 留 5 GB 以上显存；超出上限的组排队，卡位空出自动补位。
- 调度脚本 `demo_env/launch_variants.sh`（`start|status|log <v>|stop`，原子锁防重复起），探活脚本 `demo_env/runs_probe.sh`：按 `/proc/<pid>/cwd` 区分副本、按 `CUDA_VISIBLE_DEVICES` 统计卡位。**只统计真正的 python 进程**——`/isaac-sim/python.sh` 是 shell 包装脚本，它自身也匹配 `scripts/train.py` 且继承容器默认的可见卡，会把这些计数搞乱。
- 日志写 `demo_project/runs/_logs/<变体>.log`（**必须在副本之外**：副本根目录出现未跟踪文件，`pre_learn` 的脏树检查会拒绝启动），检查点与 TensorBoard 留在副本自己的 `logs/rsl_rl/recovery_agile_01/`。
- 初态库复用规则：库缓存键含源模型指纹、USD 内容、关节顺序、物理步长/分频、`num_envs` 与采集参数、`collection/core/asset_cfg/env_cfg` 的实现哈希，但**不含奖励权重**。因此 6 组（两个基线、`reward_*`、`ppo64`、`ppo_explore`）共用同一份 2048-env 库（缓存命中），改增益、控制率、初态参数的 5 组各自重采（每组约 5 min）。
- 实测 9 组并行时 2048 envs 约 7~8 s/iteration（单卡独占 5.5 s），整批约 10 h。

### 1.5 评估与基线设计

- 评估用 `scripts/petbot_eval.py`：**确定性策略**（`act_inference`，无探索噪声）、`split=eval` 初态库（采集 seed 与训练不同）、512 episodes、按初态分类（直立 / 倒置 / 侧躺）拆分。
- 另设两个**地板基线**，用同一批 512 初态判定上面的数字意味着什么：恒定零动作（`demo_env/zero_action_eval.py`）、恒定保持默认构型（动作 `-0,-0.5,-0.5,0,0,0,0`，即把 5 个关节指令到默认角、后轮零速）。
- CLI 覆盖边界：Hydra **只接受 runner（agent）侧**覆盖（`agent.*`）；env 字段不被接受（`rewards.still.weight=…` 直接报 `Key 'rewards' is not in struct`），而 `decimation`、`episode_length_s` 又在 `__post_init__` 里赋值、覆盖顺序不可靠。因此**奖励 / 控制率 / 回合长度的变体一律做成代码副本**（改完在副本内 commit 保持工作区干净），只有 actor-critic 参数走 `agent.*`。

## 2. 执行现象

### 2.1 时间线

| 时间 | 事件 |
|---|---|
| 10-13 19:19 | 首次铺 4 组失败：副本缺派生资产（`FileNotFoundError: Prepare the asset`），且日志写在副本里 |
| 10-13 19:23 | 补资产后重铺，`reward_still` / `reward_config_heavy` 因 Hydra 拒绝 env 字段而秒退 |
| 10-13 19:26 | 修正前铺开 9 组；为避免脚本被改动后按错位字节执行误杀训练进程，调度器被强制终止 |
| 10-13 20:39 | 用修正版脚本重起调度器：已在跑的 5 组 + `episode15`/`ppo64` 跳过，重起两个 reward 组（改为代码副本） |
| 10-14 01:00 / 02:10 | 卡位空出，`spawn1m`、`ppo_explore` 先后补位 |
| 10-14 08:38 | 11 组全部跑满 3000 iteration |
| 10-14 09:00–09:20 | 11 组独立评估（3 路并行）+ 两个基线 |

### 2.2 训练期现象

并行 9 组时各组 iteration time 6.2~9.4 s（`ppo64` 因 rollout 加倍约 14~18 s）；每 100 iteration 存一个检查点，共 31 个/组。

**训练期 `Recovery/success_rate`（末 100 iteration 均值）几乎全是 0**：

| 变体 | 训练期 | 确定性评估 |
|---|---|---|
| `base_s43` | 0.175 | 0.225 |
| `base_s42` / `gains2x` / `gains_down` / `ppo64` | 0.000 | 0.227 |
| `reward_config_heavy` | 0.000 | 0.084 |
| `spawn1m` | 0.000 | 0.053 |
| `reward_still` / `episode15` / `decimation5` / `ppo_explore` | 0.000 | 0.000 |

同一检查点在训练期是 0.000、在确定性评估下是 0.227，说明**训练期这个指标测的是"带探索噪声的策略"**，不能用于组间比较。

### 2.3 独立评估现象（确定性策略，512 episodes）

| 变体 | 成功率 | 侧躺类 | 中位成功用时 | 末态稳定 | episode 时长 p50 |
|---|---|---|---|---|---|
| `base_s42` / `gains2x_s42` / `gains_down_s42` / `ppo64_s42` | 0.227 | 0.249 | 1.42–1.48 s | 0.21–0.23 | 10.0 s |
| `base_s43` | 0.225 | 0.247 | 1.48 s | 0.213 | 10.0 s |
| `reward_config_heavy_s42` | 0.084 | 0.094 | 3.10 s | 0.018 | 5.0 s |
| `spawn1m_s42` | 0.053 | 0.059 | 2.62 s | 0.006 | 5.3 s |
| `reward_still` / `episode15` / `decimation5` / `ppo_explore` | 0.000 | 0.000 | — | 0.000 | 7.5–15 s |

- 分类：`直立` 与 `倒置` 类**全部为 0**（倒置类只在 `spawn1m` 出现 0.025），成功全部落在"侧躺/倾斜"类。
- 两个基线（同一批 512 初态）：**恒定零动作 0.000**；**恒定保持默认构型 0.213**（侧躺类 0.234）。
- 初态库构成（训练库与评估库几乎一致）：直立（±15°）0.0~0.8%、高度在容差内约 60%、**构型在容差内 0.000**、低速约 94%，姿态上 88% 属"侧躺/倾斜"。
- `reward_config_heavy`、`spawn1m` 及 4 个 0 分组的**中位 episode 时长明显短于上限**（5.0~7.6 s < 10 s），即大量 episode 因越界 / 非法状态提前结束——策略在乱动。

### 2.4 机制结论

1. **"不动"就能拿分**：约 21% 的初态本来就满足成功判据（零动作不行，但保持默认构型不动就行）。因此 `Recovery/success_rate` 与 `stable` 奖励（权重 5.0，最大正项）都能被"不动"拿到 —— 与"必须发生恢复"的目标直接冲突。**修法**：给成功判据与奖励加门控，例如只在"初始未满足条件"的样本上计分、按初态分类统计、排除初态即满足条件或近直立的样本。
2. **探索噪声挡住了"站定"**：训练动作是采样的（`init_noise_std=0.7`，未见衰减），"所有条件连续保持 1 s"在噪声下几乎不可达 → 训练期该指标恒 0（唯一例外 `base_s43` 0.175），而确定性策略能到 0.225~0.227。**修法**：评估一律用确定性策略（已如此）；训练侧给噪声做衰减课程或降低 σ，否则 `stable` 的 `held` 项在训练中不可得。
3. **伺服增益不是当前瓶颈**：kp/kd ×0.5、×2 与默认落在同一水平（0.227）；`decimation5`（100 Hz）、`episode15`、`reward_still` 反而更差（0.000），更大的探索噪声（`ppo_explore`）也是 0，与第 2 条一致。

### 2.5 实现层面的坑（都已处理，供下一轮直接套用）

| 现象 | 原因 | 处理 |
|---|---|---|
| 训练启动即 `FileNotFoundError: Prepare the asset…` | 每个副本都需要自己的派生资产，只复制代码不够 | 已备好的资产拷进各副本 `artifacts/petbot_asset/`；改了增益/模型的组各自 `prepare`（串行，不与其它 Kit 实例并发） |
| `RuntimeError: Commit the demo changes…` | 日志写在副本根目录 → 未跟踪文件 → 脏树 | 日志统一写 `runs/_logs/` |
| `Could not override 'rewards.still.weight' … Key 'rewards' is not in struct` | Hydra 只认 agent 侧覆盖 | env 字段改代码副本并 commit |
| 并发计数翻倍 | `/isaac-sim/python.sh` 包装进程也匹配 `scripts/train.py` | 探活只统计 exe 为 python* 的进程 |
| 评估进程静默退出、无 traceback | `petbot_eval.py` 入口块 `try: main() finally: app.close()`，`app.close()` 会直接结束进程 | 用 `demo_env/run_eval_debug.py` 包装运行（打印 traceback 后 `os._exit`） |
| 评估报 `Incorrect type under namespace: /seed` | `env_cfg.from_dict(params/env.yaml)` 类型严格匹配，`seed: 42` 撞上默认 `None` | `petbot_eval.py` 增加 `restore_cfg`：按报错逐项丢弃无法还原的顶层键（会丢掉 `seed` 与整个 `scene` 段；评估跑在同一代码副本里，`scene` 默认值一致，当前无影响） |
| 多个 Kit 实例共用 `cache/kit` 的 key-value 锁警告 | 共享缓存 | 无害，忽略 |
| 容器以 root 运行 → `artifacts/`、`logs/` 在宿主属 root | 3.0.0-beta2 镜像默认 uid 1000 写不了属主为宿主的 checkout | 已知；宿主侧删除需 `docker exec … rm -rf` |

### 2.6 产物位置

- 运行副本（11 个，被 `demo_project/.gitignore` 排除）：`demo_project/runs/<变体>/`
- 日志：`demo_project/runs/_logs/<变体>.log`；评估日志：`runs/_logs/eval_<变体>.log`、`hold_default_base_s42.log`、`zero_action_base_s42.log`
- 检查点（31 个/组，最后为 `model_2999.pt`）：`runs/<变体>/logs/rsl_rl/recovery_agile_01/<时间>_petbot/`
- 评估结果：`runs/<变体>/artifacts/petbot_eval.json`（含每个 episode 的初态分类、成功标记、用时）；基线：`runs/base_s42/artifacts/{zero_action_eval,hold_default_eval}.json`
- 工具：`demo_env/{launch_variants.sh,runs_probe.sh,eval_all.sh,eval_report.py,zero_action_eval.py,run_eval_debug.py}`

## 3. 对下一轮的约束

1. **先修信号，再谈预算**：加"必须恢复"的门控（判据 + `stable` 奖励），并给探索噪声做衰减课程/降低 σ；否则加 iteration、换超参都只会把"不动"学得更稳。
2. 训练预算提到 10000+ 时，比较仍用确定性评估 + 按初态分类，并始终带上地板基线（零动作、保持默认构型）作为参照。
3. 已知无效的方向（本轮已排除）：伺服增益 0.5×~2×、控制率 100 Hz、回合 15 s、单纯加重静止/动作率/力矩惩罚、更大的探索噪声。
4. 值得下一轮试的方向：按"初态分类"给不同权重的起身塑形（例如倒置/侧躺类才有高度与姿态进展奖励）、在初态库里剔除"已满足条件"的样本、或参考 AGILE 的"提升辅助 + 课程"思路但只对真正倒地的样本启用。
