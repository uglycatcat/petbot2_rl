# 参考项目 recovery：试训记录

本文是参考项目 `recovery` 这条线的**完整记录**：结论 → 判据 → 实验 → 环境 → 产物与查询 → 踩过的坑 → 参数速查。进度主线见 [AGENTS.md](../AGENTS.md#进度报告文档主线)，上游来源见 [reference_project/README.md](../reference_project/README.md)。

更新日期：2026-10-12。

## 一句话结论

1. `recovery` 是已调研项目里**唯一**同时满足「轮足 + Isaac Lab + 倒地起身 + SO(3) 全姿态随机 + 零参考动作」的 —— 因此被选为当前重点试训对象。
2. 训练 / 评估 / 渲染链路**全部跑通**；上游原配置跑满 10000 iteration 后**完全失败**（论文口径成功率 0.00%），第一轮 6 个论文口径实验同样 C = 0.00%。
3. 失败原因已定位到**实现与论文的语义差异**（课程权重 CW 方向反了、episode 长度、奖励组合），**不是**判据问题、不是训练量问题。
4. **上游仓库只有 Thunder 机器人，没有论文的 Go2-W / KYON**；已把作者的 Go2-W 模型接进 Isaac Lab 并启动论文标准训练，同时用作者公开权重在 MuJoCo 上取得**复现基准**（官方权重确实能从倒地起身站住）。
5. 当前判断：**瓶颈主要在被训平台本身**（Thunder 49.3 kg + 硬伺服），而非论文方法 —— Go2-W 上同一套论文口径的早期学习信号显著更好。

## 项目信息

| 项 | 值 |
|---|---|
| 上游 | https://github.com/Kitjesen/recovery |
| 路径 | `reference_project/recovery`（浅克隆 depth 1；`repo sync -g recovery` 物化） |
| commit | `5b57477598963695be27474c0ba941a1ba4afd38`（2026-04-17） |
| 我们的修改 | 已推到本仓库分支 **`refproj/recovery`**（上游 + 补丁 + `PATCHES.md`）；`default.xml` 的 recovery 项指向它 |
| 许可证 | Apache-2.0（Thunder URDF / 网格由 Qiongpei Technology 授权随包分发） |
| 论文 | Deng et al., *Learning to Recover: Dynamic Reward Shaping with Wheel-Leg Coordination for Fallen Robots*，[arXiv:2506.05516](https://arxiv.org/abs/2506.05516)（v2，2025-10-08） |
| 机器人（仓库自带） | **Thunder**：18 关节 = 12 revolute（4 腿 × hip/thigh/calf）+ 4 continuous（轮，命名 `*_foot_joint`）+ 2 fixed（lidar/camera）；总质量 49.325 kg |
| 机器人（论文用的） | **Go2-W**（Unitree，19.13 kg / 16 关节）与 **KYON**（IIT，未公开）。**两者都没有随 recovery 仓库发布**；Go2-W 可从作者部署仓库 `boyuandeng/Recovery_go2w` 取得（见下） |
| 任务 | `RobotLab-Isaac-Velocity-Recovery-Thunder-v0`（method1_deng）、`...-Thunder-Getup-v0`（method2_getup，未跑）；Go2-W 任务由我们仿建 |
| 依赖 | 仅 Isaac Lab + RSL-RL，不依赖 robot_lab |

## 复现判定口径（三套判据）

| 判据 | 定义 | 用途 |
|---|---|---|
| **C（论文平地口径）** | 末步高度 > 0.42 m **且** 12 个腿关节偏差 ≤ 0.5 rad **且** 最大关节速度 < 0.1 rad/s 持续 1 秒，在 2048 个随机倒地初态上评估 | 判定"是否复现论文"，论文报告 97.8%(Go2-W) / 99.1%(KYON) |
| A（上游原版） | 与 C 同，但把 4 个 continuous 轮关节也算进偏差与速度 | 复现上游自带判据 |
| B | 与 C 同，只看腿关节 | 分离"轮关节污染"的影响 |

评估脚本 `eval_recovery.py` 三套一起输出，自动识别 episode 长度与关节数，并按初始倒地姿态（直立 / 侧躺 / 斜躺 / 倒置）分层。

**关于 0.42 m 的适用范围**：论文那 0.42 m 出现在 KYON 的章节，**不能套到其他平台**。实测：Go2-W 在作者 PD（Kp 25 / Kd 0.5）下保持默认姿态的稳态基座高度是 **0.299 m**（留档 `recovery_env/go2w_stance_probe.out`），作者策略末态高度 p50 = 0.323 m、p90 = 0.374 m；用 0.42 m 判定实测成功率为 **0.00%**。评估 Go2-W 时用"平台自适应判据"：高度 > 92% 该平台实测站姿高度（0.299 m）+ 倾角 < 20° + 腿关节偏差 ≤ 0.5 rad + 最后 1 s 最大关节速度 < 0.1 rad/s。

## 上游实现的两个缺陷（已定位）

1. **`recovery_success_rate` 是 1e-6 权重的日志项**：加权后永远打印 `0.0000`，**不能用训练日志判断成败**，必须用评估脚本。论文 README 里"在 TensorBoard 看成功率爬到 0.97"在实际日志中不可读。
2. **判据把 4 个 continuous 轮关节算进"偏差 < 0.5 / 速度 < 0.1"**：轮关节无位置限位、绝对角度无限累积，理论上不可满足 —— 这解释了 A 判据恒为 0。在 6144 个样本的离线评估里，这两项通过率是**精确 0.0%**（同批样本中，高度 > 0.30 m 通过 29.7%、姿态对齐 20.5%）。

另外仓库 README 的奖励权重表与代码实际权重不一致（代码多一个 `recovery_joint_deviation` 项）；上游 README 称 critic 观测约 262 维，实测 **285 维**（actor 78 维与 README 一致）。**以代码为准**。

## 论文与实现的语义差异（第一轮的根因）

逐条对照论文原文（§III-A/B/C/D/E + Eq.1~4 + Table I + §IV-A）后确认的差异，按影响排序：

| # | 项 | 上游实现 | 论文 | 影响 |
|---|---|---|---|---|
| 1 | **课程权重 CW** | `CW(i) = 0.3 × 0.968^i`，**随 iteration 衰减**（iter 35 → 0.096、iter 100 → 0.0116、iter 1000 → 2.3e−15，行为惩罚在训练早期就全部关闭） | Eq.3：`cw ← cw^β`，β = 0.968，**每次 episode 重置前更新** → 从 0.3 **递增到 1**（iter 35 → 0.68、iter 100 → 0.955、iter 200 → 0.998；"早期快速学到任务行为，后期动作更平滑"） | **最关键**：终局奖励里碰撞 / 动作变化率 / 关节速度 / 力矩 / 关节加速度**全为 0** —— 惩罚被关掉，策略"站得住但还在晃"（关节速度 p50 7.63 rad/s） |
| 2 | episode 长度 | 5 s | §IV-A："Each episode lasts **5 seconds** (at a control rate of 50Hz)" | 第一轮曾误改成 4 s，已改回 5 s |
| 3 | 行为惩罚的相乘项 | `CW × DS` | Eq.4：`r_t = ed·(r_joint_pos + r_base_height + r_base_ori) + cw·(r_collision + r_action_rate + r_joint_vel + r_torque + r_joint_acc) + r_wheel_vel` —— **行为组只乘 CW，不含 DS** | 中 |
| 4 | 腿动作缩放 | hip 0.125 / thigh·calf 0.25 | 论文只给形式 `s_p·a + q`（数值未给）；作者部署栈用 **0.5**（腿轮均 0.5） | 大（探索幅度只剩 1/2~1/4） |
| 5 | 行为惩罚权重 | 全部比论文低 5~7 倍 | Table I：collision −5e−2、action_rate −1e−2、joint_vel −2e−2、torque −2.5e−5、accel −2.5e−7、wheel_vel −2e−2 | 中 |
| 6 | 姿态项形式 | 有界 `exp(−‖g_b−e_z‖²)` × 50 | Table I：**二次 `(g_b−e_z)²`** 权重 50（倒置时代价可达 ~200，梯度更强） | 中（实测换回有界 exp 更差） |
| 7 | 论文没有的项 | 多一个 `joint_deviation` 惩罚；`support_state` 带 0.30 m 高度门 | 论文只要求"**四轮同时触地**"作为支撑状态，无高度门；无 `joint_deviation` | 小 |
| 8 | DR | 摩擦 U[0.3,1.0]；执行器增益 ×(0.85,1.15)；有周期推力与 reset 外扰 | §IV-A：基座质量 ±5 kg、连杆 ±10%、摩擦 U[**0.7,1.3**]、刚度/阻尼 perturbed by **±1.0 / ±0.1**、动作噪声 **σ=0.02** | 小~中 |

论文没给数值的地方，取**作者自己的部署实现**（`Recovery_go2w/simulate_python/test/runtest.py`：动作 `a×0.5 + q_default`、腿 `Kp 25 / Kd 0.5`、轮 `Kd 0.1`、50 Hz），不自行发明。

**动作噪声 σ=0.02 已实现**（自定义 action term，`noisy_actions`，上游无此接口）；观测噪声沿用上游实现的量级（角速度 ±0.2、关节速度 ±1.5、其余 ±0.05/±0.01），论文只说"高斯、对关节速度与基座速度更大"未给具体值。

## 实验矩阵与实测结果

统一配置：`--max_iterations 10000 --headless`，seed 42（复现性实验换种子），Thunder 任务 50 Hz 控制；**Thunder 用 4096 envs，Go2-W 用 2048 envs**（4096 下 Go2-W 的网格碰撞体会让 PhysX 初始化失败）。

资源实测（2026-10-12 在 6 路并发下测）：单 run 约 **4.4~4.9 个核**（`450%~490%` CPU）、**2.6~5.0 GB 显存**；两卡各约 14.4 GB / 24.6 GB（利用率 53%~57%）。迭代耗时随并发数与同机他人负载波动：**单跑 4.42 s**（4096 envs，历史实测均值）、6 路并发时 **Thunder(4096 envs) 6.6~10.6 s、Go2-W(2048 envs) 6.8~7.4 s**。曾试过 8 路并发（当时 load 达到 ~99、迭代从 6.6 s 涨到 11~12 s，聚合吞吐反而下降）→ **以 6 路为准**。

### 第一代：上游原配置（跑满 10000 iter）—— 失败基线

| 指标 | 值 |
|---|---|
| 严格成功率 A / B | 0.00% / 0.00% |
| 高度 + 姿态达标 | 20.90% |
| 分初态：直立 / 侧躺 / 斜躺 / **倒置** | 77.9% / 12.96% / 17.70% / **0.00%** |
| 末态高度中位数 | 0.203 m（Thunder 站立需 0.426 m） |
| 腿关节偏差 / 最大速度中位数 | 0.762 rad / 1.118 rad/s |

**学会"保持站立"，不会从倒地起身。** 已排除"判据的锅"（只看腿关节仍 0）与"训练量不足"（跑满 19.7 亿样本，奖励 iter 99 后趋平）。开销：日志 10000 次 iteration time 均值 4.42 s → ≈44.4k steps/s、总时长 ≈12.3 h；51 个检查点。

### 第二代：论文口径 6 个并行实验（第一轮，全部跑满）

| 实验 | 与论文口径的差异 | C | 站姿达标 | 直立 / **倒置** / 侧躺 | 腿偏差 p50 | 腿速度 p50 |
|---|---|---|---|---|---|---|
| **`paper_v1`** | 8 项已改；动作噪声未实现 | 0.00% | **43.16%** | 65.2 / **46.75** / 38.21 | 1.722 | 7.627 |
| `v3_long_ep` | episode 8 s | 0.00% | 17.97% | 83.3 / 0.00 / 8.64 | **0.257** | **0.871** |
| `v2_exp_ori` | 姿态项回有界 exp | 0.00% | 20.70% | 86.4 / 0.00 / 11.96 | 0.302 | 1.172 |
| `v6_100hz` | 控制 100 Hz | 0.00% | 14.06% | 53.0 / 1.30 / 8.97 | 2.274 | 3.055 |
| `v5_soft_servo` | 腿伺服 25 / 0.5 | 0.00% | 3.32% | 22.7 / 0.00 / 0.00 | 1.768 | 2.149 |
| `v4_scale_only` | 上游原版 + 仅腿缩放 0.5 | 0.00% | 0.20% | 0 / 0 / 0.33 | 0.879 | 1.588 |

两个关键发现：
1. **只有 `paper_v1` 学会从倒地起身**（倒置 46.75%、侧躺 38.21%），其它变体在倒置初态基本 0% → "改回论文口径"确实买到了起身能力。
2. **"会起身"与"能站静"在错误的 CW 下互相冲突**：`paper_v1` 起身能力升（倒置 29.9%→46.8%，iter 2000→9999）的同时静止性持续变差（腿速度 4.50→7.63）；而 `v2`/`v3` 站得静（速度 0.87~1.17）却完全不会起身（倒置 0%）。这正是 CW 被实现成"衰减"（惩罚被关掉）的指纹。

已被早期数据否掉的两个假设：① "论文的二次姿态惩罚太狠，换有界的更好" → `v2` 明显更差；② "伺服太硬是瓶颈" → `v5`（软伺服）反而最差。

### 第三代：论文标准口径（进行中）

差异 = 表「论文与实现的语义差异」全部改成论文原义，并补上动作噪声 σ=0.02。包含：Eq.3 递增 CW、Eq.4 组合、5 s @ 50 Hz、Table I 权重与二次姿态项、§IV-A 的 DR、删除论文没有的项（`wheel_leg_coord` 全删；`support_state` 去掉高度门，另留一门控对照）。

| 实验 | 平台 | 说明 | 进度（2026-10-12 21:00 实测） |
|---|---|---|---|
| `p1_paper_exact` | Thunder | 论文标准，seed 42 | 3295 / 10000 |
| `p3_paper_exact_s123` | Thunder | 论文标准，seed 123 | 3252 / 10000 |
| `p4_paper_baseline_noED` | Thunder | 论文 §IV-B 的 Baseline（DS 关闭） | 3219 / 10000 |
| `w3_cw_rise_5s` | Thunder | 上一代延续：5 s + 递增 CW（无动作噪声） | 3717 / 13200（续训后目标，见坑 15） |
| **`g1_go2w_paper`** | **Go2-W** | 论文机器人 + 论文标准口径，2048 envs，seed 42 | 2846 / 10000 |
| **`g2_go2w_paper_s7`** | **Go2-W** | 同上，seed 7 | 2791 / 10000 |

中途评估（论文口径 C 仍 0.00%，属该进度下的正常区间）：

| 指标 | `p1_paper_exact` @2600（Thunder） | `g1_go2w_paper` @2000（**Go2-W**） | 作者官方权重（Go2-W 终局，MuJoCo） |
|---|---|---|---|
| 站姿达标 | 10.16% | **25.20%** | — |
| h42 通过率 | 10.6% | **37.5%** | — |
| 末态基座高度 p50 | 0.238 m | **0.407 m** | 0.323 m |
| 姿态误差 p50 | 1.470 rad | **0.133 rad** | — |
| 腿关节偏差 p50 | 0.873 rad | 0.919 rad | 0.485 rad |
| 最后 1 s 最大关节速度 p50 | 10.60 rad/s | **0.834 rad/s** | **0.010 rad/s** |

**读法**：同一套论文口径，**Go2-W 上的早期信号明显优于 Thunder**（站姿达标 2.5 倍、姿态误差小一个数量级、关节速度低 12 倍），且其末态高度已超过官方权重在 MuJoCo 里的水平。这支持"瓶颈在被训平台而非论文方法"的判断。两者距离"站静"都还差，需等 6000~8000 iteration 的曲线。

## 论文平台（Go2-W）接入与官方权重验证

上游恢复仓库**没有**论文用的机器人模型。Go2-W 从作者部署仓库取得：

| 项 | 值 |
|---|---|
| 来源 | `reference_project/Recovery_go2w`（`boyuandeng/Recovery_go2w@e6424625`，在 `default.xml` 清单里，目前只在训练服务器物化） |
| 模型（上游原件） | `unitree_robots/go2w/go2w.xml` + `assets/` 网格（作者自己使用的 MuJoCo 模型） |
| 模型（实际使用的副本） | 训练 / 验证都走副本：`reference_project/recovery_variants/g1_go2w_paper/assets/go2w/{go2w.xml,scene.xml}`；转出的 USD 在同目录 `assets/usd/`；权重副本 `assets/model_7999.pt` 与 `recovery_env/models/model_7999.pt` |
| 权重（上游原件） | `simulate_python/test/log/model_7999.pt`（作者在 Go2-W 上训练的策略，`iter=7999`；**网络结构已实测核对**：actor `78 → [128,128,128] → 16`、critic `103 → [128,128,128] → 1`，留档 `recovery_env/go2w_checkpoint_shapes.out`）—— **许可证未核实，不进本仓库** |
| 机器人数值 | `nq/nv/nu = 23/22/16`（1 个浮动基座关节 + 16 个驱动关节 = 4 腿 × hip/thigh/calf/wheel）；总质量 **19.1264 kg**；力矩上限来自 MJCF 的 `<motor ctrlrange>`（hip/thigh 23.7、calf 45.43、轮 15；`forcerange` 未设）—— 数值留档 `recovery_env/go2w_model_facts.out` |
| 部署契约（`runtest.py`） | 动作 `a×0.5 + q_default`、腿 `Kp 25 / Kd 0.5`、轮 `Kd 0.1`、策略 50 Hz；obs 布局 `16+3+3+12+12+4+12+12+4`（与我们 Thunder 任务同构） |

**官方权重在 MuJoCo（作者 MJCF + 作者推理契约，32 个随机倒地初态）上的实测**：

| 指标 | p10 | **p50** | p90 |
|---|---|---|---|
| 末步倾角 | 4.2° | **11.3°** | 24.3° |
| 末步基座高度 | 0.243 m | 0.323 m | 0.374 m |
| 腿关节偏差 | 0.362 | 0.485 rad | 0.950 |
| 最后 1 s 最大关节速度 | 0.002 | **0.010 rad/s** | 7.50 |

→ **论文的官方策略确实能自己翻身并站稳**；按"平台自适应判据"成功率 **37.5%**（直起初态 44% / 侧躺 30% / 倒置 38%）。这是本机自己训练时要对标的**复现基准**。
演示视频（MuJoCo 离屏渲染，需 `MUJOCO_GL=egl`）：`reference_project/recovery_variants/g1_go2w_paper/logs/go2w_policy_ep{0,1,2}.mp4` 与拼图 `go2w_demo_preview.png`。

**Go2-W 在 Isaac Lab 里的接入**：作者 MJCF → USD（Isaac Lab `MjcfConverter`），修掉三处转换问题：① 导入器多带一个 articulation root（`worldBody`）→ 解析不到唯一 articulation；② 刚体缺 `ContactReportAPI` → 接触传感器报错；③ prim 多一层包装（`/Robot/base_link/...`）导致路径模式不匹配。4096 envs 下 Go2-W 的网格碰撞体让 PhysX 初始化失败 → **2048 envs 稳定**。

## 运行环境（训练服务器）

| 项 | 值 |
|---|---|
| 宿主 | Ubuntu 22.04（kernel 5.15），32 核 64 线程，125 GB RAM |
| 显卡 | 4 × RTX 4090 24 GB（**共用机器**，0/1 号长期有他人负载；实验只用 2/3 号） |
| 容器 | `recovery-train`（训练）、`recovery-vis`（观察 / 渲染 / 评估），同镜像同挂载，`--gpus all -e CUDA_VISIBLE_DEVICES=2,3 --shm-size 32g`；容器内以 root 运行（产物属主为 root） |
| 镜像 | `nvcr.io/nvidia/isaac-lab:2.3.2`；Isaac Lab 2.3.2 / Isaac Sim 5.1 / Python 3.11.13 / `rsl-rl-lib 3.1.2` |
| 挂载 | `reference_project/recovery`、`recovery_variants`、`recovery_env` → 容器内 `/workspace/*`，**所有写入都落在 `reference_project/` 内** |

### 网络注意（踩过，别重犯）

- `docker pull` **默认必然失败**：dockerd 被强制走 `HTTP_PROXY=127.0.0.1:7890`（他人配置，该端口原本无服务）。解法：在 7890 上起一个只做直连隧道的代理（`recovery_env/direct_proxy_7890.py`，用 `env.sh proxy start` 启停），之后 `docker pull` 正常；**重启或新会话后需要重开**。
- 大镜像用 `parallel_pull.py`（多连接分块 + sha256 + 生成 docker-load tarball）：镜像站 `nvcr.m.daocloud.io` 约 5 MB/s，直连 `nvcr.io` 仅 0.7 MB/s。`crane` 单连接仅约 1 MB/s 且在该镜像站上协议报错，不要用。
- `pypi.org` 与 `raw.githubusercontent.com` 直连不通 → 用 `pypi.tuna.tsinghua.edu.cn` / `gh-proxy.com`。

### 渲染能力（已验证）

- Isaac Sim 离屏渲染可用：`render_test.py` 出 90 帧 / 854×480 mp4（首帧亮度 113.6、首末帧差 13.01）；训练脚本自带 `--video --video_length 200 --video_interval 2000` + `--enable_cameras` 可录 1280×720 / 50 fps。
- **MuJoCo 离屏渲染**需 `MUJOCO_GL=egl`（默认会报 "an OpenGL platform library has not been loaded"）。
- 尚未用 `--video` 录过我们自己训练策略的演示（`scripts/play.py` 未跑）；目前策略效果靠指标判断。

### 兼容性缺口：`dump_pickle`

`scripts/train.py` 导入的 `isaaclab.utils.io.dump_pickle` 在 Isaac Lab 2.3.2 不存在 → 用标准库 `pickle` 补一个同名函数。这是上游唯一**必需**的源码改动，已随 `refproj/recovery` 分支保存（`patches/upstream_dump_pickle.patch`）。

## 产物清单与怎么查

### 版本管理内（本仓库）

| 位置 | 内容 |
|---|---|
| 分支 `refproj/recovery` | 上游 recovery **+ 我们的补丁**（`PATCHES.md` + `patches/`）；`default.xml` 指向它 |
| 分支 `refproj/recovery-tools` | **我方试训工具箱**：脚本、补丁、每个实验的完整定义（`variant_diffs/*.patch`）、关键评估结果 JSON —— 见该分支的 `tools/recovery_trial/INDEX.md` |
| `docs/reference_recovery_validation.md` | 本文 |

### 工作目录（训练服务器，`reference_project/` 下，不进版本管理）

| 路径 | 内容 |
|---|---|
| `reference_project/recovery/` | 上游 checkout（已应用论文口径补丁）+ 训练产物 |
| `reference_project/recovery_env/` | 全部脚本与工作入口（见下）；`_images/isaac-lab-2.3.2.tar`（7.9 GB）镜像已 load，**无用可删** |
| `reference_project/recovery_variants/<代号>/` | 各实验的独立 checkout + 日志 / 检查点（`<代号>` 含义见 `variant_diffs/` 与 INDEX.md） |
| `reference_project/Recovery_go2w/` | 作者部署栈（Go2-W 模型与官方权重） |

主要脚本（`reference_project/recovery_env/`）：

| 文件 | 作用 |
|---|---|
| `env.sh` | 环境入口：`fetch / up / install / verify / proxy / train / vis / status / down / shell` |
| `eval_recovery.py` | **评估入口**：判据 A/B/C、自动识别 episode 长度、按初态分层 |
| `eval_variant.sh` | 对某实验最新检查点做一次论文口径评估：`./eval_variant.sh p1_paper_exact 512 1` |
| `eval_batch.py` | 批量评估与汇总表（`--parse-only` 重算已有日志） |
| `launch_variants.sh` | 启动 / 重启实验 |
| `summary.sh` | 各实验进度与奖励趋势一屏（清单需与实验一起更新；日志带 ANSI 转义，脚本内已剥离） |
| `watchdog.sh` + `watchdog.log` | 巡检 / 自动续训 / 定期评估脚本（每 tick 记录一行进度）。**当前未启用**（用户决定不常驻）：因此没有自动续训与定期自动评估，崩溃的训练需要手动处理（`./watchdog.sh once` 做一轮巡检并自动续训，`./watchdog.sh start` 才常驻） |
| `go2w_policy_check.py` | 官方权重在 MuJoCo 上的验证（含录像） |
| `go2w_model_facts.py` + `.out` | Go2-W 模型客观数值留档：总质量 19.1264 kg、力矩上限（`<motor ctrlrange>` 23.7 / 45.43 / 15）、`nq/nv/nu = 23/22/16` |
| `go2w_checkpoint_shapes.py` + `.out` | 作者权重网络结构留档：actor `78→[128,128,128]→16`、critic `103→[128,128,128]→1`、`iter=7999` |
| `go2w_stance_probe.out` | Go2-W 默认姿态在作者 PD 下的稳态站姿高度实测留档（**0.299 m**，即 0.42 m 判据对该平台不适用的依据）。注：该探针只跑 1 episode，仅用于测站姿高度；成功率数字以 32 episodes 的 `go2w_check3.json` 为准 |
| `convert_go2w.py` / `fix_go2w_usd.py` | Go2-W MJCF → Isaac Lab USD 转换与修复 |
| `parallel_pull.py` / `direct_proxy_7890.py` | 镜像拉取与 docker 代理绕行 |
| `render_test.py` / `make_demo_preview.py` | 离屏渲染验证 / 演示拼图 |
| `paper_vs_impl.md` | 论文与实现偏差清单（含第一轮 8 项修正说明） |

常用命令（在 `reference_project/recovery_env/` 下）：

```bash
./summary.sh                          # 当前实验的进度与奖励趋势（一屏）
./eval_variant.sh g1_go2w_paper 512 1 # 立刻评估某个实验（论文口径 C 判据）
./launch_variants.sh <代号>            # 启动/重启某实验（脚本内含 GPU/seed/env 数）
./watchdog.sh once                    # 一轮巡检：发现死掉的实验就用最新检查点自动续训
./watchdog.sh status                  # 只有在常驻启用时才有意义（当前不启用）
./env.sh shell train                  # 进训练容器
```

注意事项：**评估必须用与训练同一份代码**（`eval_variant.sh` 自动设 `PYTHONPATH`）；**不要并发跑评估**（多个 Isaac Sim 实例同卡会 OOM，实测三个 1024 envs 并发即失败，串行每个 512 envs 约 3 GB）；**看门狗当前不常驻**，所以主机重启或崩溃后没有任何自动恢复：训练进程要手动用 `./launch_variants.sh <代号>` 拉起（**会从头开始**），想从最近检查点续训用 `./watchdog.sh once`（它走 `--resume`）；**不要用日志里的 `recovery_success_rate` 判成败**（恒为 0.0000）。

## 踩过的坑

1. **并发启动训练会竞争 URDF→USD 转换** → `ValueError: No contact sensors added to the prim ... no rigid bodies are present`。启动要错开 20 s 以上，失败就单独重启那一个。
2. **重建容器会丢 editable 安装** → `ModuleNotFoundError: No module named 'thunder_recovery'`。补 `pip install -e .`；变体用 `PYTHONPATH` 不受影响。
3. **同卡并发评估 → CUDA OOM**（见上）。
4. **并发训练上限 6 路**：8 路时 load 99/64，迭代 6.6 s → 11~12 s，聚合吞吐下降。
5. **录制视频**：`--video_interval 1` 只出 1 帧 → 用 `--video_length 200 --video_interval 2000`；视口会同时拍到所有并排 env，单机演示要把相机对准 `env_0`。
6. 容器日志里的 `omni.kit.pipapi` traceback 是 Kit 想联网装扩展，**无害**；`Disabling key-value database because another kit process is locking it` 也是无害警告。
7. `render_test.py` 在 `--enable_cameras` 下 `app.close()` 会挂住 → 用 `os._exit()`。
8. `eval_recovery.py` 踩过：`torch.inference_mode()` 下策略对 obs 原地更新报 `RuntimeError`（改 `no_grad`）；`torch.full(dtype=object)` 不支持；numpy / torch 比较要先转换。
9. `pkill -f "<模式>"` 会匹配到自己 → 模式写成 `direct_proxy_7890[.]py` 这类形式。
10. URDF 细节容易看错：Thunder 的轮关节命名是 `*_foot_joint`，判据要**按关节类型筛**而不是按名字。
11. 容器内以 root 跑训练 → 产物属主是 root（要改运行身份需停一次训练）。
12. **浅克隆（depth 1）不能推送到完整仓库**：`remote unpack failed: index-pack failed`。先 `git fetch --unshallow` 再推。
13. **Go2-W USD 转换三处修复**（多 articulation root / 缺 ContactReportAPI / prim 层级）见上一节；4096 envs 会因网格碰撞体初始化失败，用 2048。
14. **MuJoCo 无头渲染**要 `MUJOCO_GL=egl`。
15. **`rsl_rl` 的 `--resume` 下 `--max_iterations` 是"再加多少迭代"**，不是在总数上截断：`w3_cw_rise_5s` 从 `model_3200.pt` 续训后目标是 **13200**（3200 + 10000），不是 10000。续训时想控制总长要按剩余量传参。
16. **训练日志带 ANSI 颜色转义**（`^[[1m` / `^[[0m`）：按字段位置解析会取错列（`summary.sh` 曾因此把迭代数显示成 `iteration/10000`）。正确做法是先 `sed -r 's/\x1b\[[0-9;]*[A-Za-z]//g'` 剥掉，再按 `数字/数字` 这种 token 匹配。
17. **看门狗可能被静默杀掉**（日志无报错、无退出码；实测 2026-10-12 正常跑 7 个 tick 后消失）。它当前不常驻启用；若某次启用了，要定期 `./watchdog.sh status` 确认还活着。

## 关键参数速查

**Thunder 资产**：`assets/thunder/urdf/thunder.urdf`（SolidWorks 导出，URDF 名 `thunder`）；总质量 49.325 kg（base_link 18.25 kg）；另一份 `thunder_0131copy.urdf` 未被使用。

**论文关键设定（§III / §IV-A，已核对原文）**：
- 初态与自由落体：SO(3) 均匀随机姿态 + 关节角扰动，从 **1.1 m 自由落体 2 s、零力矩**，仅超时终止。
- episode：**5 s @ 50 Hz**（250 控制步）。
- ED（episode-based dynamic shaping）：`ed = ((a·t)/T)^k`，**a = T/2**，k = 3 —— 自由落体结束时 ed = 1，episode 末尾 ed 达 8。
- CW（课程权重）：**`cw ← cw^β`，β = 0.968，每次 episode 重置前更新，从 0.3 递增到 1**（不是衰减！）。
- 奖励组合（Eq.4）：`r_t = ed·(r_joint_pos + r_base_height + r_base_ori) + cw·(r_collision + r_action_rate + r_joint_vel + r_torque + r_joint_acc) + r_wheel_vel`。
- Table I 权重：`joint_pos 42`、`base_height 120`、`base_orientation 50`（形式 `(g_b − e_z)²`）、`collision −5e−2`、`action_rate −1e−2`、`joint_vel −2e−2`、`torque −2.5e−5`、`accel −2.5e−7`、`wheel_vel −2e−2`。关节位与高度是 **exp 形式**。**h\* = 0.42 m 是 KYON 的量级，不适用于其他平台**（Thunder 用 0.426 m，Go2-W 站姿仅 0.30~0.36 m）。
- PPO：asymmetric actor-critic，lr 1e-3、γ = 0.99；MLP `[128,128,128]`。
- DR（§IV-A）：基座质量 ±5 kg、连杆 ±10%、摩擦 U[0.7,1.3]、刚度/阻尼 perturbed by ±1.0 / ±0.1、**动作噪声 σ = 0.02**。

**Go2-W（论文平台，作者部署栈）**：`a×0.5 + q_default`（腿轮均 0.5）；腿 `Kp 25 / Kd 0.5`、轮 `Kd 0.1`；策略 50 Hz；16 关节 / 19.13 kg；actor `78 → 16`、MLP `[128,128,128]`；obs 布局 `16+3+3+12+12+4+12+12+4`。

**PetBot 与参考项目的区别（需求基线已明确）**：PetBot 默认站姿本来就是四轮着地的静稳定姿态，"站住"不需要学；参考项目里为学会站立付出的塑形、课程与辅助力**不能直接照搬**，其成功判据（含 0.1 rad/s 静止性）也不一定适用于 PetBot。

**PetBot 移植的前置条件**（本阶段不做）：动作必须是 **5 个关节位置目标（`head` / `left_ear` / `right_ear` / `left_frame` / `right_frame`）+ 2 个后轮速度目标**；PetBot 缺原生位置伺服；左右前轮被动；头 / 耳 / 车架均可触地支撑，不能沿用"非足端触地即失败"。

## 场景外记录（已做，无参考价值）

在桌面机（RTX 2080Ti，**不承担训练**）上曾跑过 512 envs × 400 iteration 的部分运行与一次 2048 envs 吞吐探测，**无参考价值**，仅存档。桌面机上唯一有用的结论是渲染路径不可用（显卡低于 Isaac Sim 5.1 标称要求），效果验证走 MuJoCo。

## 下一步

1. 等第三代实验（`p1`/`p3`/`p4`/`w3` 与 Go2-W 的 `g1`/`g2`）到 6000~8000 iteration，看 C 判据是否拉起；**以 Go2-W 为准对标官方权重的 37.5%（平台自适应口径）**。
2. **若拉起** → 取最好检查点录单机 demo 视频 → 交付 demo；随后按 `docs/engineering_handoff.md` 进入 PetBot 移植（换资产 + 7 维动作契约 + 位置伺服）。
3. **若 Go2-W 仍为 0** → 说明问题不在已列出的语义差异，下一轮候选：奖励尺度 / 归一化（论文公式使 reward 为大负数，可能影响 PPO 价值学习）、value loss 与 KL 是否发散、延长到 20000 iteration、或核对作者训练配置（作者训练代码未公开，只能靠部署栈反推）。
4. Thunder 线的结论已足够：**它不适合作为复现论文的平台**（49.3 kg + 硬伺服，同口径下行为显著差于 Go2-W）；若后续要它，需单独设计奖励而非照搬论文。
5. 中期可做：把我们的工具箱从备份分支提升到主仓库受版本管理的位置（当前在 `refproj/recovery-tools`）。

## 未决 / 未验证

- C 判据里"关节速度 < 0.1 rad/s"在 Thunder 上从未被满足过（当前最好 3.7 rad/s）；Go2-W 上中途为 0.834 rad/s，未达但接近一个数量级内。
- 论文的 97.8% / 99.1% 是在它自己的两台机器人上得到的；**KYON 模型未公开，Go2-W 的训练配置未公开**，本机能复现的上限受平台与未知超参限制。
- Go2-W 任务是我们仿建的（作者只公开了 MuJoCo 部署栈），奖励 / 观测 / 执行器是按论文与作者部署参数对齐，未与原训练环境逐位核对。
- 作者权重 `model_7999.pt` 的许可证未核实，因此**不进本仓库**。

## 相关文档

- [参考项目索引](../reference_project/README.md)：其余上游项目的用途与快照版本
- [当前需求基线](current_scope.md)、[工程实施说明](engineering_handoff.md)：PetBot 恢复任务的需求与设计
