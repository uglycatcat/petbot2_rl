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
