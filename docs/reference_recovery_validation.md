# 参考项目 recovery：试训记录

本文是参考项目 `recovery` 这条线的**完整记录**：结论 → 判据 → 实验 → 环境 → 产物与查询 → 踩过的坑 → 参数速查。进度主线见 [AGENTS.md](../AGENTS.md#进度报告文档主线)，上游来源见 [reference_project/README.md](../reference_project/README.md)。

更新日期：2026-10-10。

## 一句话结论

1. `recovery` 是已调研项目里**唯一**同时满足「轮足 + Isaac Lab + 倒地起身 + SO(3) 全姿态随机 + 零参考动作」的 —— 因此被选为当前重点试训对象。
2. 训练 / 评估 / 渲染链路**全部跑通**，但**上游原配置跑满 10000 iteration 后完全失败**：论文口径成功率 0.00%，没有复现论文的 97.8% / 99.1%。
3. 定位到上游实现与论文的 **8 项偏差**，按论文口径改后启动 **6 个并行实验**（1 个论文口径基线 + 5 个变体，进行中）。早期指标已反超旧配置的终局，方向初步验证。

## 项目信息

| 项 | 值 |
|---|---|
| 上游 | https://github.com/Kitjesen/recovery |
| 路径 | `reference_project/recovery`（浅克隆 depth 1） |
| commit | `5b57477598963695be27474c0ba941a1ba4afd38`（2026-04-17） |
| 许可证 | Apache-2.0（Thunder URDF / 网格由 Qiongpei Technology 授权随包分发） |
| 论文 | Deng et al., *Learning to Recover: Dynamic Reward Shaping with Wheel-Leg Coordination for Fallen Robots*，[arXiv:2506.05516](https://arxiv.org/abs/2506.05516) |
| 机器人 | Thunder：18 关节 = 12 revolute（4 腿 × hip/thigh/calf）+ 4 continuous（轮，命名 `*_foot_joint`）+ 2 fixed（lidar/camera）；总质量 49.325 kg。**论文评测用的 Go2-W / KYON 模型没有随仓库发布**，Thunder ≠ PetBot |
| 任务 | `RobotLab-Isaac-Velocity-Recovery-Thunder-v0`（method1_deng）、`...-Thunder-Getup-v0`（method2_getup，未跑） |
| 依赖 | 仅 Isaac Lab + RSL-RL，不依赖 robot_lab |

## 复现判定口径（三套判据）

| 判据 | 定义 | 用途 |
|---|---|---|
| **C（论文口径）** | 末步高度 > 0.42 m **且** 12 个腿关节偏差 ≤ 0.5 rad **且** 最大关节速度 < 0.1 rad/s 持续 1 秒，在 2048 个随机倒地初态上评估 | 判定"是否复现论文"，论文报告 97.8%(Go2-W) / 99.1%(KYON) |
| A（上游原版） | 与 C 同，但把 4 个 continuous 轮关节也算进偏差与速度 | 复现上游自带判据 |
| B | 与 C 同，只看腿关节 | 分离"轮关节污染"的影响 |

评估脚本 `eval_recovery.py` 三套一起输出，并能自动识别 episode 长度。

## 上游实现的两个缺陷（已定位）

1. **`recovery_success_rate` 是 1e-6 权重的日志项**：加权后永远打印 `0.0000`，**不能用训练日志判断成败**，必须用评估脚本。
2. **判据把 4 个 continuous 轮关节算进"偏差 < 0.5 / 速度 < 0.1"**：轮关节无位置限位、绝对角度无限累积，理论上不可满足 —— 这解释了 A 判据恒为 0。在 6144 个样本的离线评估里，这两项的通过率是**精确 0.0%**（同批样本中，高度 > 0.30 m 通过 29.7%、姿态对齐 20.5%）。

另外仓库 README 的奖励权重表与代码实际权重不一致（代码多一个 `recovery_joint_deviation` 项）；上游 README 还称 critic 观测约 262 维，实测构建出来是 **285 维**（actor 78 维与 README 一致）。**以代码为准**。

## 实验矩阵与实测结果

统一配置：`--num_envs 4096 --max_iterations 10000 --seed 42 --headless`，task = `...-Recovery-Thunder-v0`，50 Hz 控制。单 run 约吃 **10 核 / 5.5 GB 显存 / 10 GB 内存**；6 路并发时 load ≈ 35–45/64、GPU 利用率 50–84%（实测平衡点，再加会互相拖慢）。迭代约 6.6 s（`v6_100hz` 因每控制步物理步数减半，约 5.5 s）。

### 旧配置（上游原版，跑满 10000 iter）—— 失败基线

| 指标 | 值 |
|---|---|
| 严格成功率 A / B | 0.00% / 0.00% |
| 高度 + 姿态达标 | 20.90% |
| 分初态：直立 / 侧躺 / 斜躺 / **倒置** | 77.9% / 12.96% / 17.70% / **0.00%** |
| 末态高度中位数 | 0.203 m（站立需 0.426 m） |
| 腿关节偏差 / 最大速度中位数 | 0.762 rad / 1.118 rad/s |

**结论：该配置学会了"保持站立"，但不会从倒地状态起身。** 已排除"判据的锅"（只看腿关节仍为 0）与"训练量不足"（跑满论文原配置 19.7 亿样本，奖励在 iter 99 后趋平）。

跑满 10000 iteration 的实测开销：44,440 steps/s，计时钟 12 h 16 min，落盘 51 个检查点（每 200 iteration 一个）。

### 论文口径 `paper_v1`（进行中，中途评估）

| 指标 | `paper_v1` @800 | 旧配置终局 @9999 |
|---|---|---|
| 严格成功率 C | 0.00% | 0.00% |
| 高度 + 姿态达标 | **33.11%** | 20.90% |
| 倒置初态能站 | **14.19%** | 0.00% |
| 侧躺初态能站 | **26.45%** | 12.96% |
| 直立初态能站 | 81.62% | 77.9% |

C 仍为 0 的原因：末态腿关节速度中位数 3.696 rad/s（要求 < 0.1），即"站住了但还在晃"。论文自身的成功率曲线也在 6000~8000 iteration 才拉起，当前进度（~8%）属正常区间。

### 变体横向对比（同口径中途评估）

| 实验 | 与论文口径的差异 | 检查点 | 严格成功率 C | 高度 + 姿态 |
|---|---|---|---|---|
| **`paper_v1`** | 无（8 项已按论文改；动作噪声 σ=0.02 未实现） | 800 | 0.00% | **33.11%** |
| `v2_exp_ori` | 姿态项换回上游有界 exp | 1000 | 0.00% | 19.82% |
| `v3_long_ep` | episode 8 s | 1444 | — | — |
| `v4_scale_only` | 上游原版 + 仅腿动作缩放 0.5 | 1200 | 0.00% | 10.55% |
| `v5_soft_servo` | 腿伺服刚度 25 / 阻尼 0.5 | 600 | 0.00% | 2.54% |
| `v6_100hz` | 控制 100 Hz（论文 §IV-A 原文） | 912 | — | — |

**已被早期数据否掉的两个假设**：① "论文的二次姿态惩罚太狠，换有界的更好" → v2 明显更差；② "伺服太硬是瓶颈" → v5（软伺服）反而最差。

### 8 项偏差修正（`paper_repro_v1.patch`，4 文件 +96/−72）

腿动作缩放 0.125/0.25 → 0.5；塑形改论文式 `DS = ((a·t)/T)^k`（a = T/2, k = 3）；行为惩罚权重恢复论文 Table I；姿态项改 `(g_b − e_z)²` 权重 −50；按论文 Eq.4 重排 DS/CW；删除上游自加的 `joint_deviation` 项；DR 按论文 §IV-A；episode 5 s → **4 s**（保持 50 Hz）。保留了上游的 `support_state`（0.30 m 高度门）与 `wheel_leg_coord` 门控。

**唯一仍与论文不同的地方**：论文 §IV-A 要求的动作噪声 σ=0.02 没有实现 —— Isaac Lab 的动作项没有这个接口，需要自定义 action term。

## 运行环境（训练服务器）

| 项 | 值 |
|---|---|
| 宿主 | Ubuntu 22.04（kernel 5.15），32 核 64 线程，128 GB RAM |
| 显卡 | 4 × RTX 4090 24 GB（**共用机器**，0/1 号长期有他人负载；实验只用 2/3 号） |
| 容器 | `recovery-train`、`recovery-vis`，同镜像同挂载，`--gpus all -e CUDA_VISIBLE_DEVICES=2,3 --shm-size 32g`，容器内以 root 运行（产物属主为 root） |
| 镜像 | `nvcr.io/nvidia/isaac-lab:2.3.2`；Isaac Lab 2.3.2 / Isaac Sim 5.1 / Python 3.11.13 / `rsl-rl-lib 3.1.2` |
| 挂载 | `reference_project/recovery`、`recovery_variants`、`recovery_env` → 容器内 `/workspace/*`，**所有写入都落在 `reference_project/` 内** |

### 网络注意（踩过，别重犯）

- `docker pull` **默认必然失败**：dockerd 被强制走 `HTTP_PROXY=127.0.0.1:7890`（他人配置，该端口原本无服务）。解法：在 7890 上起一个只做直连隧道的代理（`recovery_env/direct_proxy_7890.py`，用 `env.sh proxy start` 启停），之后 `docker pull` 正常；**重启或新会话后需要重开**。
- 大镜像用 `parallel_pull.py`（多连接分块 + sha256 + 生成 docker-load tarball）：镜像站 `nvcr.m.daocloud.io` 约 5 MB/s，直连 `nvcr.io` 仅 0.7 MB/s。
- `pypi.org` 与 `raw.githubusercontent.com` 直连不通 → 用 `pypi.tuna.tsinghua.edu.cn` / `gh-proxy.com`。

### 渲染能力（已验证）

服务器的离屏渲染链路可用：以最小交互场景渲染 90 帧、854 × 480，输出 `render_test.mp4`（86 KB），首帧亮度均值 113.6（非全黑）、首末帧平均差 13.01（画面确实在变）；demo 录制链路也实测通过（1280 × 720 / 50 fps，抓帧确认机器人正常渲染）。

**尚未录制 recovery 策略本身的演示**（`scripts/play.py` 未跑），所以策略效果目前只能靠指标判断。脚本见 `recovery_env/render_test.py`。

### 兼容性缺口：`dump_pickle`

`scripts/train.py` 导入的 `isaaclab.utils.io.dump_pickle` 在 Isaac Lab 2.3.2 不存在 → 用标准库 `pickle` 补一个同名函数（上游唯一源码改动，已存档 `upstream_dump_pickle.patch`）。

## 产物清单与怎么查

脚本与工作目录都在 `reference_project/recovery_env/`（该目录不进版本管理）：

| 文件 | 作用 |
|---|---|
| `env.sh` | 环境入口：`fetch / up / install / verify / proxy / train / vis / status / down / shell` |
| `parallel_pull.py` | 多连接镜像拉取器（分块续传 + sha256 + 生成 docker-load tarball） |
| `direct_proxy_7890.py` | 127.0.0.1:7890 直连隧道代理（绕开 dockerd 的死代理） |
| `verify.py` | 容器内自检（版本、依赖 API、Isaac Sim 启动、task 注册） |
| `eval_recovery.py` | **评估入口**：三套判据 A/B/C，自动识别 episode 长度，按初始姿态分层输出 |
| `eval_variant.sh` | 对某变体最新检查点做一次论文口径评估：`./eval_variant.sh paper_v1 512 1` |
| `launch_variants.sh` | 启动 / 重启变体训练（全部或指定单个） |
| `summary.sh` | 六实验进度与奖励趋势一屏汇总 |
| `watchdog.sh` + `watchdog.log` | **训练看门狗**：15 分钟巡检自动续训；每 2 小时自动评估 `paper_v1` |
| `render_test.py` | 离屏渲染验证（出 mp4） |
| `inspect_checkpoints.py` | 检查点检查工具 |
| `paper_vs_impl.md` | 论文 vs 实现 8 项偏差清单 |
| `paper_repro_v1.patch`、`upstream_dump_pickle.patch` | 补丁存档 |
| `README.md` | 环境记录 + 六实验怎么查 + 坑 |
| `_images/isaac-lab-2.3.2.tar` | 7.9 GB，镜像已 load 进 docker，**无用可删** |

训练产物：

| 路径（相对仓库根） | 内容 |
|---|---|
| `reference_project/recovery/` | 上游 checkout（含已应用的论文口径补丁）+ 训练产物（138 MB） |
| `reference_project/recovery/logs/train_paper_v1.out` | `paper_v1` 训练日志 |
| `reference_project/recovery/logs/train_4096x10000.out` | 旧配置训练日志 |
| `.../logs/rsl_rl/thunder_recovery/2026-10-10_05-13-30_paper_v1/` | `paper_v1` 检查点（每 200 迭代一个） |
| `.../logs/rsl_rl/thunder_recovery/2026-10-09_10-42-27/` | 旧配置检查点，含 `model_9999.pt` |
| `.../logs/eval_recovery.json`、`eval_paper_v1*.json` | 评估结果 JSON |
| `reference_project/recovery_variants/<v>/` | 五个变体的独立 checkout + 各自日志 / 检查点（86 MB） |
| `reference_project/Recovery_go2w/` | 作者仓库（含 `model_7999.pt`、`simulate_python/test/runtest.py`，242 MB） |

常用命令（`<仓库根>/reference_project/recovery_env` 下）：

```bash
./summary.sh                        # 六实验进度与奖励趋势一屏
./watchdog.sh status                # 看门狗是否在跑 + 最近进度
./watchdog.sh log 50                # 看门狗日志（含自动续训与定期评估记录）
./eval_variant.sh paper_v1 512 1    # 立刻做一次论文口径评估
./launch_variants.sh v3_long_ep     # 重启某个变体
./env.sh shell train                # 进训练容器
```

注意事项：

- **评估必须用与训练时同一份代码**（`eval_variant.sh` 已自动设置 `PYTHONPATH`）；变体之间口径不同（episode 长度、伺服增益、控制频率），数字只有在同一份代码下才可比。
- **不要并发跑评估**：多个 Isaac Sim 实例同卡同时起会把显存挤爆（实测三个 1024 envs 并发 → `CUDA out of memory`）。串行跑，每个 512 envs 约 3 GB。
- **主机重启后**要手动 `./watchdog.sh start` 和 `./launch_variants.sh`（容器不会自动重启，训练也不会）。
- 训练日志里**不要**用 `Episode_Reward/recovery_success_rate` 判断成败（1e-6 权重的日志项，恒为 `0.0000`）；要看 `Learning iteration` 是否推进，并用 `eval_recovery.py` 评估。

## 踩过的坑

1. **并发启动训练会竞争 URDF→USD 转换** → 报 `ValueError: No contact sensors added to the prim ... no rigid bodies are present`。启动要错开（20 s 以上），失败就单独重启那一个实验。
2. **重建容器会丢掉 editable 安装** → `ModuleNotFoundError: No module named 'thunder_recovery'`。补 `pip install -e .`；变体因用 `PYTHONPATH` 不受影响。
3. **多个 Isaac Sim 实例同卡并发评估 → CUDA OOM**（见上）。
4. **录制演示视频**：`--video_interval 1` 会产出只有 1 帧的 mp4 → 用 `--video_length 200 --video_interval 2000`；视口会同时拍到所有并排的 env，单机演示要把相机对准 `env_0`。
5. 容器启动日志里的 `omni.kit.pipapi` traceback 是 Kit 想联网装扩展，**无害**；`Disabling key-value database because another kit process is locking it` 也是无害警告。
6. `render_test.py` 在 `--enable_cameras` 下 `app.close()` 会挂住 → 用 `os._exit()`。
7. `eval_recovery.py` 踩过：`torch.inference_mode()` 下策略对 obs 原地更新报 `RuntimeError`（改 `no_grad`）；`torch.full(dtype=object)` 不支持；numpy / torch 比较要先转换。
8. `pkill -f "<模式>"` 会匹配到自己 → 模式写成 `direct_proxy_7890[.]py` 这类形式。
9. URDF 细节容易看错：轮关节命名是 `*_foot_joint`，判据要**按关节类型筛**而不是按名字。
10. 容器内以 root 跑训练 → 产物属主是 root（要改运行身份需停一次训练）。

## 关键参数速查

**Thunder 资产**：`assets/thunder/urdf/thunder.urdf`（SolidWorks 导出，URDF 名 `thunder`）；总质量 49.325 kg（base_link 18.25 kg）；另一份 `thunder_0131copy.urdf` 未被使用。

**论文关键设定**：episode 4 s、控制 100 Hz（400 控制步）；自由落体 1.1 m / 2 s / 零力矩，仅超时终止；`DS = ((a·t)/T)^k`，`a = T/2`，`k = 3`；课程权重 `CW = 0.3 × 0.968^i`；PPO lr 1e-3、γ 0.99。

**Table I 奖励权重**：`joint_pos 42`、`base_height 120`（目标 `h* = 0.42 m`）、`base_orientation 50`（形式 `(g_b − e_z)²`）、`collision −5e−2`、`action_rate −1e−2`、`joint_vel −2e−2`、`torque −2.5e−5`、`accel −2.5e−7`、`wheel_vel −2e−2`。

**DR（§IV-A）**：基座质量 ±5 kg、连杆 ±10%、摩擦 U[0.7, 1.3]、刚度 / 阻尼 ±15 / ±2；**动作噪声 σ=0.02 未实现**（Isaac Lab 的动作项无此接口，需自定义 action term —— 已记入 `paper_vs_impl.md`）。

**作者部署栈（`runtest.py`）**：动作 `a × 0.5 + q_default`（腿轮均 0.5）、`Kp 25 / Kd 0.5`、策略 50 Hz；actor `78 → 16`，MLP `[128,128,128]`；obs 布局 `16+3+3+12+12+4+12+12+4`。

**PetBot 移植的前置条件**（本阶段不做）：动作必须是 **5 个关节位置目标（`head` / `left_ear` / `right_ear` / `left_frame` / `right_frame`）+ 2 个后轮速度目标**；PetBot 缺原生位置伺服；左右前轮被动；头 / 耳 / 车架均可触地支撑，不能沿用"非足端触地即失败"。

## 场景外记录（已做，无参考价值）

在桌面机（RTX 2080Ti，**不承担训练**）上曾跑过 512 envs × 400 iteration 的部分运行与一次 2048 envs 吞吐探测。**这些记录无参考价值**，仅存档说明"做过"：该机不用于训练，只在最后做 MuJoCo 等效果验证。桌面机上唯一有用的结论是渲染路径必崩（显卡低于 Isaac Sim 5.1 标称要求），所以效果验证要走 MuJoCo。

## 下一步

1. 等 `paper_v1` 跑到 6000~8000 iteration（约 2026-10-11 凌晨），看严格成功率 C 是否拉起。
2. **若拉起** → 取最好检查点录单机 demo 视频 → 交付 demo，随后进入 PetBot 移植（换资产 + 7 维动作契约 + 位置伺服）。
3. **若仍为 0** → 下一轮候选：奖励尺度 / 归一化（论文公式使 reward 为大负数，可能影响 PPO 价值学习）、补上论文的动作噪声 σ=0.02、检查 value loss 与 KL 是否发散、或延长到 20000 iteration。
4. `v3_long_ep` 的 DS 尺度与 episode 长度耦合（8 s 末端 DS 涨到 64），若 v3 表现好，下一轮改用与 episode 解耦的 DS 形式。

## 未决 / 未验证

- C 判据里"关节速度 < 0.1 rad/s"在任何配置上都还没被满足过（当前最好也只有 3.7 rad/s）。
- 论文的 97.8% / 99.1% 是在它自己的两台机器人上得到的；**本机可复现的上限可能受 Thunder 这个平台（49.3 kg、4 轮）的实际能力限制**。

## 相关文档

- [参考项目索引](../reference_project/README.md)：其余上游项目的用途与快照版本
- [当前需求基线](current_scope.md)、[工程实施说明](engineering_handoff.md)：PetBot 恢复任务的需求与设计
