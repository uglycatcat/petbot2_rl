# PetBot demo 服务器工具（tools/petbot_trial）

本目录是 `demo_project/recovery_agile_01` 在训练服务器上使用的**容器与实验工具**的版本化副本，按 `AGENTS.md` 规则 13 的 `refproj/*-tools` 惯例备份：分支 `demoproj/recovery_agile_01-tools`，属于我方工具而非上游项目，**不写进 `default.xml`**。设计与现象记录见 demo 分支的 [PETBOT_SWEEP_01.md](../../PETBOT_SWEEP_01.md)。

## 用法

- 服务器上运行时它们放在 `<repo>/demo_project/demo_env/`；从本分支取用后把本目录内容拷到那里即可（脚本向上查找 `default.xml` 定位仓库根，放两处都能用）。
- 顺序：`./env.sh up` → `./env.sh install` → `./env.sh prepare`（派生资产）→ `./launch_variants.sh start`（并行扫描）→ `./eval_all.sh`（确定性评估）→ `python3 eval_report.py`（汇总表）。地板基线用 `zero_action_eval.py`。

```bash
# 例：铺一轮扫描（先确认变体目录已备好资产）
PETBOT_MAX_ITERS=3000 ./launch_variants.sh start     # 后台跑：terminal background / setsid
./launch_variants.sh status                          # 卡位、进度、iteration time
./eval_all.sh && python3 eval_report.py              # 11 组确定性评估 + 汇总
```

## 文件

| 文件 | 作用 |
|---|---|
| `env.sh` | 容器 `petbot-train` 的建/启/装/验证/prepare/训练/shell。**关键**：挂载必须复刻宿主相对层级（demo → `/workspace/demo_project/recovery_agile_01`，并挂 `/workspace/.repo/projects/demo_project`、`/workspace/.repo/project-objects`），否则容器内 git 不可用，而 `pre_learn` 开头的脏树检查会让训练直接失败 |
| `launch_variants.sh` | 并行扫描调度：变体→卡位映射、每卡并发上限（GPU3 ≤ 6、GPU2 ≤ 3）、超限排队自动补位、原子锁防重复起 |
| `runs_probe.sh` | 容器内探活：按 `/proc/<pid>/cwd` 认副本、按 `CUDA_VISIBLE_DEVICES` 认卡位；**只统计真正的 python 进程**（`/isaac-sim/python.sh` 包装进程会污染计数） |
| `eval_all.sh` | 逐个跑确定性评估（`split=eval` 初态库、512 episodes），3 路并行 |
| `eval_report.py` | 汇总各组 `artifacts/petbot_eval.json`：总成功率、按初态分类、成功用时分布、末态稳定 |
| `zero_action_eval.py` | 地板基线：恒定动作（默认全 0；`--action=-0,-0.5,-0.5,0,0,0,0` 即"恒定保持默认构型"的开环基线） |
| `library_stats.py` | 初态库构成统计（直立/高度/构型/低速通过率）——判断"成功率里有多少来自本来就站得住的状态" |
| `run_eval_debug.py` | 包装 `scripts/petbot_eval.py` 运行：其入口 `try: main() finally: app.close()` 会吞掉 traceback 并静默退出，用它才能看到异常 |
| `direct_proxy_7890.py` | dockerd 被强制走 `127.0.0.1:7890` 时用的直连代理（拉镜像前 `./env.sh proxy start`） |
| `debug/debug_jointnames.py`、`debug/debug_geom_rename.py` | 关节名被加 `_1` 后缀问题的诊断与修复验证（`Sdf.BatchNamespaceEdit` 先让开同名 geom prim，再把关节改回契约名） |

## 约定与坑（细节见 PETBOT_SWEEP_01.md）

- 每个运行副本都需要**自己的派生资产**（只拷代码不够）；改了增益/模型/控制率/初态参数的副本各自 `prepare`，且**不要并发跑 `prepare`**。
- 变体日志必须写在副本**之外**（`demo_project/runs/_logs/`）：副本根目录出现未跟踪文件会被 `pre_learn` 的脏树检查拒绝。
- CLI 覆盖只对 runner（agent）侧有效（`agent.*`）；env 字段（`rewards.*`、`decimation`、`episode_length_s`）必须做**代码副本**并在副本内 commit（保持工作区干净）。
- 容器以 root 运行 → 副本里的 `artifacts/`、`logs/` 在宿主侧属 root，删除需经 `docker exec`。

## 第二轮训练与配对评估（2026-10-14）

`sweep2_pipeline.py` 使用干净 demo HEAD 创建 8 个固定版本副本，GPU3 五组、GPU2 三组；每组 2048 envs × 3000 iteration。启动间隔至少 45 s，实际空闲显存不足时排队。首组完整训练初态库生成后复用，失败不自动重试。独立评估使用同一份 512 初态及状态张量 hash，比较默认构型、零动作、固定序列及策略的 750/1500/2999 检查点；汇总单独报告救回开环失败样本与丢失开环成功样本。

从主仓库运行 `python3 demo_project/demo_env/sweep2_pipeline.py --prepare-only` 检查并准备副本；正式运行去掉 `--prepare-only`，用当前具有 Docker 权限的会话启动独立后台进程。用 `--status` 查看进度并更新汇总。输出在 `demo_project/runs/_sweep2/`：`status.json`、训练/评估日志、`summary.json`、`SUMMARY.md`。已有状态或进程锁会阻止重复启动。检查点和 TensorBoard 在各副本 `logs/rsl_rl/recovery_agile_01/`。

服务器的 systemd user manager 未继承当前 shell 的 Docker 补充组，直接由它执行 docker 会报 socket permission denied；本轮已保存该次失败记录，改由已有 Docker 权限的 shell 派生独立进程，没有修改系统权限。状态文件须结合实际进程、日志和检查点核查。

## 复位姿态诊断（2026-10-15）

`sweep2_sequence_xyzw_diagnostic.py` 基于训练代码 `9553fef` 的评估入口，独立使用 Lab 3 根姿态 **xyzw** 计算初始重力，打印缓存/直接计算的差异与倒地数量。仅用于排查，未修复训练任务中的课程与复位观测。运行时从 demo checkout 执行，设置 `PYTHONPATH` 为该 checkout；参数同 `petbot_sweep2_eval.py`，本次为 `--episodes 512 --mode sequence --dataset /workspace/runs/_sweep2/eval_states.pt`，输出 `sequence_xyzw_diagnostic.json`。已确认缓存 fallen=0、直接计算=509；修正序列成功 35/512。详细证据在主线 `docs/petbot_sweep2_results.md`。

## 第三轮训练与门槛（2026-10-15）

`sweep3_pipeline.py` 创建 3 组 × 2 种子的固定版本副本，2048 环境，暂跑至 1500 iteration；GPU3 四组、GPU2 两组，实际显存门槛与 45 秒错峰启动。必须先有 `_sweep3_precheck/gate.json` 的真实通过记录，且代码与测试版本相同（允许仅文档提交）。不自动续训或重启失败任务。评估 250、750、1499 检查点，复用旧独立评估库但显式确认 xyzw 契约，重新测 default、zero、初始策略基线；输出 `_sweep3/`。操作参数 `--prepare-only` / `--status`，正式启动不带这两个选项。1500 之后是否延长由实测趋势决定。

`sweep3_prepare.py` 在容器内预先采集 2048 环境口径的训练库并测量可恢复倒地索引，不运行 PPO。正式副本可复用这些缓存，`pre_learn` 仍校验全部身份字段。`sweep3_gate.py --runs-root /workspace/runs --write-gate` 汇总 PhysX、概率、检查点重载及两组 100 iteration 短训的真实证据；未全部通过时拒绝写入启动门槛文件。
