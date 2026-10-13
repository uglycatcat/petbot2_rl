# agile_env —— WBC-AGILE / `StandUp-T1-v0` 的专用容器

本目录是 **WBC-AGILE 这个参考项目专用的容器环境**，与 `recovery_env`、`robot_lab_env` **各自独立的容器，不混用**（独立的容器名、挂载点与工作目录）。宿主机不跑 Isaac Sim、不训练。

## 容器与镜像

| 项 | 值 |
|---|---|
| 容器 | `agile-train`（单容器：装依赖、跑训练、跑评估都用它） |
| 基础镜像 | `nvcr.io/nvidia/isaac-lab:3.0.0-beta2`（上游 `workflows/Dockerfile` 用的基底） |
| 拉取方式 | 镜像站多连接：`nvcr.m.daocloud.io/nvidia/isaac-lab:3.0.0-beta2`（amd64 压缩 ~16 GB，`-j 24` 实测 ~12.7 MB/s） |
| 挂载 | `reference_project/WBC-AGILE` → `/workspace/agile`；本目录 → `/workspace/agile_env`；Isaac 缓存 → `cache/`（不写 `~`） |
| GPU | `AGILE_GPU`（默认 `2,3`；0/1 号被他人任务占用） |
| 上游要求 | Isaac Lab 3.0.0-beta2 / Isaac Sim 6.0 / Python 3.12 / rsl-rl-lib 5.4.1（`pyproject.toml` + `uv.lock`） |

## 用法

```bash
./env.sh fetch        # 拉基础镜像（已存在则跳过）
./env.sh up           # 创建/启动 agile-train
./env.sh install      # 上游方式：容器内 uv sync --frozen（按 uv.lock）
./env.sh install-lite # 备选：pip install -e . --no-deps + 补非 isaaclab 小依赖
./env.sh verify       # 自检：版本、任务注册、StandUp-T1-v0 环境创建
./env.sh shell        # 进容器
./env.sh train --task StandUp-T1-v0 --num_envs 4096 --headless
./env.sh down         # 停容器（不删除）
```

注意：官方 isaac-lab 镜像里没有 `python`，容器内统一用 `/isaac-sim/python.sh`；`uv` 需在容器内 `pip install uv`（`env.sh` 会自动装）。

## 自检结果（2026-10-13 实测）

`./env.sh verify` → **PASS**（`StandUp-T1-v0` 环境创建成功）：

| 项 | 实测值 |
|---|---|
| 容器内环境 | Python 3.12.13（`/isaac-sim/python.sh`）、torch 2.10.0+cu128、isaaclab 6.1.11（= Isaac Lab 3.0.0-beta2 构建）、rsl-rl-lib 5.4.1（**已打上游补丁**，`scripts/verify_rsl_rl.py` 通过）、mujoco 3.8.1、leapp 0.5.2 |
| GPU | `torch.cuda.is_available()=True`，可见 2 张 4090 |
| 任务 | `StandUp-T1-v0` 注册成功，环境创建成功（4 envs），T1 资产从 Isaac Nucleus 正常加载 |
| 动作 | `action_dim=23`，action terms = `['joint_pos', 'lift']`（全关节位置 + 提升辅助） |
| 观测 | policy 为 5 步历史的嵌套 dict：`base_ang_vel (N,5,3)`、`projected_gravity (N,5,3)`、`joint_pos (N,5,23)`、`joint_vel (N,5,23)`、`actions (N,5,23)`；critic 另外含 `contact_forces (N,5,24)`、`base_height (N,5,1)` |
| 终止/课程 | 终止 `time_out` / `no_height_progress` / `invalid_state`；课程 6 项（含 `adaptive_lift`、`increase_action_rate_regularization` 等） |

## 装依赖的两条路（实测结论）

- `./env.sh install-lite`（**当前采用**）：`pip install -e . --no-deps` + 只补非 isaaclab 依赖（`rsl-rl-lib==5.4.1`、`leapp==0.5.2`、`mujoco` 等），约 30 秒。这样用的是**镜像自带的 torch 2.10.0 / isaaclab 6.1.11**。
- `./env.sh install`（上游方式）：`uv sync --frozen` 按 `uv.lock` 装 `torch==2.11.0` / `isaaclab[isaacsim]==3.0.0b2`，会重复下载 isaacsim（数 GB）。**未采用**，因为镜像已带同版本栈。
- **版本偏差（需知悉）**：与 `uv.lock` 的严格 pin 相比，当前是 torch 2.10.0（pin 2.11.0）、torchvision 0.25.0（pin 0.26.0）、pytest 9.1.0（pin ≥9.1.1）；`datasets` / `dearpygui` / `plotly` / `seaborn` 未装（可视化/数据类可选依赖）。如遇训练脚本报缺依赖，再按需补装即可。

## 容器内的两个坑（已处理）

1. **3.0.0-beta2 镜像默认以 `ubuntu`(uid 1000) 运行**，而挂载的 checkout 属主是宿主 `anna`(uid 1028) → 容器内写不了 `/workspace/agile`（`pip install -e` 会 `Permission denied`）。`env.sh` 已加 `--user root`（与上游 `workflows/Dockerfile` 的 `USER root` 一致）。
2. `simulation_app.close()` 会直接结束进程，**它之后的 `print` 不会输出** —— `verify.py` 已把结论行提前打印。

## 已知边界（来自源码核查，尚未实测的部分已标注）

- 上游 `StandUp-T1-v0` 是**人形全身起身**任务（Booster T1，全关节位置动作 + `LiftAction` 提升辅助），**不是混合动作**，其 T1 资产走 Isaac Nucleus（`ISAAC_NUCLEUS_DIR`），另有 `agile-download-assets` 从 `BoosterRobotics/booster_assets` 拉仿真资产 —— **这两项都需要容器内能访问外网**。
- 上游**没有发布起身预训练权重**：`agile/data/policy/` 只有 4 个 velocity 策略，且在我们的 checkout 里都是 Git LFS 指针（131–132 B），要 `git lfs pull` 才有实体。
- `agile/evaluation/specs/StandUp-T1-v0.yaml` 只有 `video_only: true` —— 起身成功判据要自己定。
