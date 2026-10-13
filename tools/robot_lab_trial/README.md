# robot_lab_env —— robot_lab v2.3.2 的专用容器

本目录是 **robot_lab 这个参考项目专用的容器环境**，与 `recovery_env`、`agile_env` **各自独立的容器，不混用**（独立的容器名、挂载点与工作目录）。宿主机不跑 Isaac Sim、不训练。

## 容器与镜像

| 项 | 值 |
|---|---|
| 容器 | `robotlab-train` |
| 基础镜像 | `nvcr.io/nvidia/isaac-lab:2.3.2`（**本机已有**，robot_lab v2.3.2 正对应 Isaac Lab 2.3.2） |
| 挂载 | `reference_project/robot_lab` → `/workspace/robot_lab`；本目录 → `/workspace/robot_lab_env:ro`；Isaac 缓存 → `cache/` |
| GPU | `ROBOTLAB_GPU`（默认 `2,3`） |
| 安装方式 | 容器内 `pip install -e source/robot_lab`（等同上游 `docker/Dockerfile` 的做法） |

上游的 `docker/Dockerfile` + `docker-compose.yaml` 是**薄封装**：基础镜像由 build arg `ISAACLAB_BASE_IMAGE` 传入（`.env.base` 里只是占位名 `isaac-lab-base`），所以这里直接复用本机已有的 2.3.2 镜像，不需要派生镜像。

## 用法

```bash
./env.sh up           # 创建/启动 robotlab-train
./env.sh install      # 容器内 pip install -e source/robot_lab
./env.sh verify       # 自检：导入、任务注册、Go2W 环境创建
./env.sh shell        # 进容器
./env.sh train --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 --num_envs 4096 --headless
./env.sh down         # 停容器（不删除）
```

## 自检结果（2026-10-13 实测）

`./env.sh verify` → **PASS**（`RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0` 环境创建成功）：

- 容器内：Python 3.11.13、torch 2.7.0+cu128、isaaclab 0.54.2、gymnasium 1.2.1、`torch.cuda.is_available()=True`（可见 2 张 4090）。
- `pip install -e source/robot_lab` 成功（robot_lab-2.3.2、v2.3.2；另从清华源补装 pinocchio / onnxruntime / cusrl 等，约 3 分钟）。
- 注册 `RobotLab-*` 任务 **50 个**；headless 创建 Go2W 环境：`num_envs=4`、policy obs `(4, 57)`、**`action_dim=16`、action terms = `['joint_pos', 'joint_vel']`** —— 正是"腿部关节位置 + 轮速"的混合动作结构。

注：在容器内**不通过 AppLauncher** 直接 `import robot_lab` 会报 `No module named 'pxr'`（USD 由 Isaac Sim 运行时提供），属正常现象，在 app 内可正常导入。
