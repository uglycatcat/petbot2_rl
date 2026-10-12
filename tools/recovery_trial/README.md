# Kitjesen/recovery 容器环境

`reference_project/recovery`（上游 commit `5b574775`）在这台服务器上的运行环境记录。
**宿主机不跑 Isaac Sim、不训练**；所有 python 都在容器里。

## 结论速览

| 项 | 状态 |
|---|---|
| 上游代码 | 已克隆到 `../recovery`（depth 1，44 个文件，含 Thunder URDF + STL） |
| 上游自带的容器配置 | **没有**。44 个文件里只有 `README.md` / `pyproject.toml` / `.gitignore` + `assets/ src/ scripts/`，无 Dockerfile / compose / 环境脚本，只能按 README 自己配 |
| 基础镜像 | `nvcr.io/nvidia/isaac-lab:2.3.2`（Isaac Lab 2.3.2 / Isaac Sim 5.1.0 / py3.11.13，17.6 GB） |
| 容器 | `recovery-train`（训练，1 个可按需扩到多卡）、`recovery-vis`（观察/录制），同一镜像、同一挂载，仅用途分离 |
| 上游安装步骤 | 已在两个容器内执行 `pip install -e .` → `thunder-recovery 0.1.0` |
| 环境自检 | 通过：Isaac Sim 可 headless 启动，两个 gym task 注册成功（见下） |
| 训练 | **进行中**：6 个实验并行（`paper_v1` + 5 个变体），4096 envs × 10000 iter，见「六个并行实验与怎么查」 |
| 上游缺口 | `scripts/train.py` 导入的 `isaaclab.utils.io.dump_pickle` 在 Isaac Lab 2.3.2 不存在 → 直接跑会 ImportError；`scripts/play.py` 本身是 TODO 桩。详见「上游待解决」 |

## 用法

```bash
cd reference_project/recovery_env
./env.sh fetch      # 拉镜像（本机 docker pull 不可用，见下）→ docker load
./env.sh up         # 创建/启动 recovery-train、recovery-vis
./env.sh install    # 容器内 pip install -e .（上游 README 的安装步骤）
./env.sh verify     # 容器内自检：版本 / 依赖 API / Isaac Sim 启动 / task 注册
./env.sh status     # 镜像与容器状态
./env.sh shell train|vis

# 上游 README 的冒烟测试（还没跑）：
./env.sh train --task RobotLab-Isaac-Velocity-Recovery-Thunder-v0 --num_envs 64 --headless --max_iterations 3 --seed 42
# 完整训练：
./env.sh train --task RobotLab-Isaac-Velocity-Recovery-Thunder-v0 --num_envs 4096 --headless --max_iterations 10000 --seed 42
# 带录制（观察容器）：
./env.sh vis --task RobotLab-Isaac-Velocity-Recovery-Thunder-v0 --num_envs 64 --headless --enable_cameras --video --video_interval 2000 --video_length 200
```

`RECOVERY_GPU` 控制可见 GPU，默认 `2,3`（0/1 号 4090 被其他任务占用）。

## 本机踩到的三个坑（都已在 env.sh 里绕过）

1. **`docker pull` 在这台机器上一定失败**：`/etc/systemd/system/docker.service.d/http-proxy.conf` 给 dockerd 设了 `HTTP_PROXY=127.0.0.1:7890`，而该端口没有服务 → 任何 registry 请求都是 `proxyconnect tcp: connection refused`。没有 root 改不了 dockerd 配置，所以走「客户端侧拉取 + `docker load`」。
2. **直连 nvcr.io 很慢**（实测 ~0.7 MB/s，8.4 GB 要 2 小时以上）：改用 `nvcr.m.daocloud.io`（NGC 国内镜像），**amd64 manifest digest 与 nvcr.io 完全一致**（`sha256:f07c37e3…`），多连接聚合 ~5 MB/s。`parallel_pull.py` 就是这个多连接拉取器（分块续传、逐层 sha256 校验、输出 `docker save` 兼容 tarball）。
3. **官方镜像里没有 `python`**：Isaac Lab 镜像的 python 是 `/isaac-sim/python.sh`（Kit 自带 py3.11），`isaaclab` 是 editable 装在 `/workspace/isaaclab/source/isaaclab`。env.sh 已统一用它。

## 镜像与容器细节

| 项 | 值 |
|---|---|
| 镜像 | `nvcr.io/nvidia/isaac-lab:2.3.2`（= `nvcr.m.daocloud.io/nvidia/isaac-lab:2.3.2`，同一 image ID `cdd5dbb5f25f`） |
| 容器内 python | `/isaac-sim/python.sh` |
| 已带依赖 | isaaclab 0.54.2、isaaclab_rl 0.4.7、isaaclab_tasks 0.11.12、rsl-rl-lib **3.1.2**（上游要求 ≥3.0.1，满足）、torch 2.7.0+cu128、numpy 1.26.0、gymnasium 1.2.1 |
| 挂载 | `../recovery` → `/workspace/recovery`（读写，训练日志落在其 `logs/`，被上游 .gitignore 覆盖）；本目录 → `/workspace/recovery_env:ro`；`cache/{ov,nvidia,kit,ov-data}` → 容器内 Isaac Sim 缓存（不写 `~`） |
| 运行参数 | `--gpus all --shm-size 32g -e CUDA_VISIBLE_DEVICES=$RECOVERY_GPU`，EULA/隐私变量已置 Y，`PIP_INDEX_URL` 指向清华源（仅兜底，本机 pypi.org 不可达） |
| 拉镜像的 tarball | `_images/isaac-lab-2.3.2.tar`（8.4 GB，装载完可删） |

## 环境自检结果（`./env.sh verify`，2026-10-09）

```
== 1. 版本 ==
python 3.11.13 / torch 2.7.0+cu128 / numpy 1.26.0 / gymnasium 1.2.1
isaaclab 0.54.2 / isaaclab_rl 0.4.7 / isaaclab_tasks 0.11.12 / rsl-rl-lib 3.1.2
== 2. 上游 train.py 依赖的 API ==
isaaclab 安装位置: /workspace/isaaclab/source/isaaclab
  isaaclab/utils/io/__init__.py 含 dump_yaml: True
  isaaclab/utils/io/__init__.py 含 dump_pickle: False   ← 上游缺口
== 3. Isaac Sim 启动 + 任务注册 ==
isaac sim 版本: 5.1.0-rc.19+release.26219.9c81211b
已注册 Recovery 任务: ['RobotLab-Isaac-Velocity-Recovery-Thunder-v0',
                      'RobotLab-Isaac-Velocity-Recovery-Thunder-Getup-v0']
Isaac Sim 启动成功
```

GPU 可被容器正常枚举（`nvidia-smi -L` 见 4×RTX 4090），Isaac Sim 启动日志里 Kit 选择了 `CUDA_VISIBLE_DEVICES` 指定的 2/3 号卡。headless 启动 1 分钟级；无 X11（`DISPLAY` 为空），所以「看画面」只能靠 `--enable_cameras` 离屏录制出 mp4。

## 上游待解决（训练前必须定）

1. **`isaaclab.utils.io.dump_pickle` 不存在**。上游 `scripts/train.py:79` 直接 `from isaaclab.utils.io import dump_pickle, dump_yaml`，第 156/157 行用它保存 `params/{env,agent}.pkl`。核实范围：
   - 本机容器内实测 Isaac Lab 2.3.2（isaaclab 0.54.2）：`io/` 只有 `__init__.py`、`yaml.py`、`torchscript.py`，全盘 `grep -rn "def dump_pickle"` 无结果；
   - PyPI（含清华镜像）上 `isaaclab` 最高 `2.3.2.post1`，其 `utils/io` 同样没有；
   - 上游 GitHub `main` 与 `v3.0.0-beta2` 的 `utils/io` 也没有 pickle 模块（`lazy_export()` 目录里无 `pickle.py`）。
   也就是说上游声明的 `isaaclab ≥ 5.0` 对应不到任何已发布版本的这个 API。可选处置（**待用户确认后再动**）：
   - a) 在上游 checkout 里做 2 行最小改动（`dump_pickle` → `pickle.dump`，或删掉这两处日志落盘），保持上游逻辑不变 —— checkout 被 `.gitignore` 覆盖，主仓库不受影响；
   - b) 换用 NGC 上最新的 `isaac-lab:3.0.0-rc1` 镜像再核实一遍（8.4 GB 重拉 + 需要重新核对其余 API）；
   - c) 暂不动上游，等上游修。
2. **`scripts/play.py` 是 TODO 桩**：不加载策略、不评估，只打印四步计划。所以要「看策略站起来」需要自己写评估入口，或走上游作者另一个仓库 `boyuandeng/Recovery_go2w`（含 `model_7999.pt` 权重与 `runtest.py`）。
3. 上游 README 的「1 × RTX 3090 / 4096 envs × 10000 iter」是它的算力声明；本机 4090 需要重新实测吞吐再定配置。

## 文件清单（本目录，全部在 `.gitignore` 覆盖范围内，不进主仓库）

| 文件 | 作用 |
|---|---|
| `env.sh` | 环境入口：fetch / up / install / verify / train / vis / status / down / shell |
| `parallel_pull.py` | 多连接镜像拉取器（分块续传 + sha256 校验 + 生成 docker-load tarball） |
| `verify.py` | 容器内自检脚本（版本、依赖 API、Isaac Sim 启动、task 注册） |
| `cache/` | Isaac Sim 运行时缓存（ov / nvidia / kit / ov-data） |
| `_images/` | 拉取镜像时的临时 tarball（镜像已 load，7.9 GB 可删） |
| `launch_variants.sh` | 启动/重启变体训练（`./launch_variants.sh` 全部 或 `./launch_variants.sh v3_long_ep` 单个） |
| `eval_variant.sh` | 对某变体最新检查点做论文口径评估：`./eval_variant.sh paper_v1 512 1` |
| `eval_recovery.py` | 评估本体（三套判据 A 上游 / B 只腿 / C 论文平地，自动识别 episode 长度） |
| `summary.sh` | 六个实验的进度与奖励趋势一屏汇总 |
| `watchdog.sh` | 训练看门狗：`start` / `stop` / `status` / `log [行数]` / `once` |
| `watchdog.log` | 看门狗日志：每 tick 一行进度（迭代数/GPU/负载），含自动续训与定期评估记录 |
| `launch_variants.sh` / `eval_variant.sh` / `summary.sh` / `watchdog.sh` | 实验与运维入口（见下节） |

## 【重要纠正】论文平台是 Go2-W / KYON，前期实验用的是第三方仓库的 Thunder

**v1~v6、w1~w4、p1~p5 共 15 个实验的训练对象都是 `Kitjesen/recovery` 自带的 Thunder，不是论文平台。**
论文 §IV-A 的两台机器人是 **KYON**（IIT 自研，未公开）和 **Unitree Go2-W**。

### 已补齐的作者资产（`reference_project/Recovery_go2w/` 内自带）

| 资产 | 位置 | 说明 |
|---|---|---|
| Go2-W 模型 | `unitree_robots/go2w/go2w.xml` + `assets/`（23 网格） | 作者做 sim-to-real 的同一份 MuJoCo 模型；16 关节（4 腿 × hip/thigh/calf/wheel），总质量 19.13 kg |
| Go2-W 权重 | `simulate_python/test/log/model_7999.pt` | 论文发布的策略，actor 78→16 [128,128,128] |
| 推理契约 | `simulate_python/test/runtest.py` | obs(78) 布局、`q_target=a*0.5+q_default`、腿 Kp25/Kd0.5、轮 Kd0.1、50 Hz、关节按类型分组 FL→FR→RL→RR |
| 注意 | —— | obs 前 16 维是**缩放后的目标位置**且按**电机顺序**排列（不是原始动作） |

### 官方权重在论文机器人上的验证（`go2w_policy_check.py`）

32 个随机倒地初态（自由落体 1.1 m / 2 s 后接管 5 s）：末步倾角 p50 **11.3°**、末高 p50 0.323 m、腿偏差 p50 0.485 rad、
最后 1 s 最大关节速度 p50 **0.010 rad/s** → **能从倒地翻身并静止站住**；按平台自适应判据
（高度 >92% 自身站姿 + 倾角<20° + 腿偏差≤0.5 + 速度<0.1）成功率 **37.5%**（直立 44% / 侧躺 30% / 倒置 38%）。

视频（需 `MUJOCO_GL=egl`）：`recovery_variants/g1_go2w_paper/logs/go2w_policy_ep{0,1,2}.mp4`、`go2w_demo_preview.png`。

### 两个必须记住的口径事实

1. 论文 "base height exceeding 0.42 m" 是 **KYON 章节**的数值；Go2-W 在这套 MJCF 里站姿高度只有 **0.30~0.36 m**，
   用 0.42 m 量 Go2-W 恒 0%（早期对 Thunder 用 0.426 m 也是照搬第三方仓库）。
2. 作者部署 PD（Kp25/Kd0.5）在 MJCF 里**撑不住默认姿态**（2 s 后腿偏差 0.413 rad、基座 0.299 m）→
   MJCF 直驱力矩模型与实机电机存在差异，sim2sim 必须实测对照。

### Go2-W 的 Isaac Lab 任务（已建好并在训练）

- MJCF→USD：`convert_go2w.py`（`MjcfConverter`，需先 `set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)`）
- USD 规范化：`fix_go2w_usd.py` —— ①只留一个 ArticulationRootAPI（导入器会产 `/go2w/worldBody` 与 `/go2w/base_link/base_link` 两个）
  ②给全部刚体加 `PhysxContactReportAPI`（接触传感器需要）
- 资产配置：`recovery_variants/g1_go2w_paper/src/thunder_recovery/config/asset_go2w.py`（腿 Kp25/Kd0.5、hip/thigh 23.7 N·m、calf 45.43、轮 Kd0.1/15）
- 任务侧改动：关节名按策略顺序、轮关节正则 `.*wheel.*`、接触体 `.*_wheel_link`、prim 路径多一层 `base_link` 包装
- **env 数上限**：4096 envs 时 PhysX 初始化失败（网格碰撞体过重）→ **2048 envs 稳定**，约 6.6 s/iter
- 实验：`g1_go2w_paper`（seed 42）、`g2_go2w_paper_s7`（seed 7），均为**论文标准口径 + 论文机器人**

### 当前 8 个训练

| 平台 | 实验 | 口径 |
|---|---|---|
| Go2-W | `g1_go2w_paper` / `g2_go2w_paper_s7` | 论文标准（§III + Table I + Eq.1/3/4 + §IV-A），2048 envs |
| Thunder | `p1_paper_exact` / `p2_…s7` / `p3_…s123` | 论文标准 |
| Thunder | `p4_paper_baseline_noED` | 论文 §IV-B Baseline |
| Thunder | `p5_paper_gated_support` | 论文 + 第三方仓库的高度门 |
| Thunder | `w3_cw_rise_5s` | 经验延续（第三方仓库口径 + 论文 CW 修正） |


## 第一轮/第二轮并行实验与怎么查（2026-10-10 起，Thunder 平台）

主机 4×4090 中 2/3 号可用，单 run 约吃 10 核 / 5.5 GB 显存 / 10 GB 内存，**6 路并发**是实测的
平衡点（GPU 利用率 50~80%、load 约 35~45/64）。全部 4096 envs × 10000 iter、seed 42。

| 实验 | 与论文口径的差异 | 容器 / 日志 |
|---|---|---|
| `paper_v1` | 无（8 项偏差全按论文改回） | recovery-train，`/workspace/recovery/logs/train_paper_v1.out` |
| `v2_exp_ori` | 姿态项用上游有界 exp 而非论文二次项 | recovery-vis，`variants/v2_exp_ori/logs/` |
| `v3_long_ep` | episode 8 s（更长恢复窗口） | 同上 |
| `v4_scale_only` | 上游原版 + 仅腿动作缩放 0.5 | 同上 |
| `v5_soft_servo` | 腿伺服刚度 25/阻尼 0.5（对齐作者部署栈 Kp/Kd） | 同上 |
| `v6_100hz` | 控制 100 Hz（论文 §IV-A 原文）+ free-fall 200 步 | 同上 |

```bash
cd reference_project/recovery_env
./summary.sh                      # 一屏看六个实验的进度与趋势
./watchdog.sh status              # 看门狗是否在跑 + 最近几行进度
./watchdog.sh log 50              # 看门狗完整日志（含自动续训、定期评估）
./eval_variant.sh paper_v1 512 1  # 立刻对 paper_v1 最新检查点做论文口径评估
```

看门狗每 15 分钟巡检一次：进程死了且未跑满 10000 迭代 → 自动用最新检查点
`--resume` 续训（该续训路径已实测通过）；每 2 小时自动发起一次 `paper_v1` 的
论文口径评估 —— 启动记录进 `watchdog.log`，**评估结果**约 5 分钟后写到
`../recovery/logs/eval_paper_v1.out`（`--out` 的 JSON 同目录）。**主机重启后需要手动
`./watchdog.sh start` 和 `./launch_variants.sh`（容器不会自动重启）。**

结果判定用 `eval_recovery.py` 的 **C 判据**（论文平地口径：末步高度 > 0.42 m +
12 个腿关节偏差 ≤ 0.5 rad + 最大关节速度 < 0.1 rad/s 持续 1 秒），与论文
97.8%(Go2-W) / 99.1%(KYON) 同口径。注意论文那两个平台的模型**没有随仓库发布**，
仓库自带的是作者自己那台 `thunder`（URDF `assets/thunder/urdf/thunder.urdf`，
18 关节 = 12 revolute + 4 continuous 轮 + 2 fixed，总质量 49.3 kg）。

