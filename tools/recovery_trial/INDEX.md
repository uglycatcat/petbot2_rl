# recovery 试训工具箱（参考分支 `refproj/recovery-tools`）

本分支保存**我方自建的试训脚本、补丁、实验定义与结果**。它们在工作时位于训练服务器的
`reference_project/recovery_env/` 与 `reference_project/recovery_variants/`（被 `.gitignore` 排除、不进版本管理），
这里是为了**避免丢失**而做的版本化备份。

- 上游 `recovery` 的修改版本身在分支 **`refproj/recovery`**（`default.xml` 指向它）。
- 本分支只含工具与实验定义，**不含**：训练日志与检查点、Isaac Sim 缓存、镜像 tar、论文作者的 Go2-W 权重
  （权重许可证未核实，需从上游 `boyuandeng/Recovery_go2w` 获取）。

## 位置与查看办法

本机（训练服务器）上的**实际使用副本**就是 `reference_project/recovery_env/`（被 `.gitignore` 排除）；这里记录的是它的**版本化副本**。工具是给 recovery 这个参考项目用的，所以放在它的 `refproj/` 命名空间下。

项目不在机器之间搬（工作区由 `git pull` 维护，参考项目由 `repo sync` 按 `default.xml` 物化），**不需要 checkout**：

```bash
git show refproj/recovery-tools:tools/recovery_trial/env.sh   # 看单个文件
git log --oneline refproj/recovery-tools                      # 看改动历史
```

## 内容

| 路径 | 说明 |
|---|---|
| `env.sh` | 环境入口（实际子命令）：`fetch / up / install / status / proxy / verify / down / train / vis / shell` |
| `go2w_model_facts.py` + `.out` | Go2-W 模型客观数值（质量 19.1264 kg、`<motor ctrlrange>` 力矩上限 23.7/45.43/15、`nq/nv/nu=23/22/16`），留档原件见 `evidence/` |
| `go2w_stance_probe.out` / `.json` | Go2-W 默认姿态在作者 PD 下的稳态站姿高度实测（0.299 m）—— 0.42 m 判据不适用于该平台的依据 |
| `parallel_pull.py` | 多连接镜像拉取器（分块续传 + sha256 → docker-load tarball）；本机 `docker pull` 默认失败（dockerd 死代理），走它或 `env.sh proxy start` |
| `direct_proxy_7890.py` | 127.0.0.1:7890 直连隧道代理，绕开 dockerd 被写死的失效代理 |
| `verify.py` | 容器内自检（版本、依赖 API、Isaac Sim 启动、task 注册） |
| `eval_recovery.py` | **评估入口**：判据 A（上游原样，含轮关节）/ B（只看腿关节）/ C（论文平地口径）；自动识别 episode 长度、按初始姿态分层、输出末态分位数 |
| `eval_batch.py` | 批量评估 + 汇总表（`--parse-only` 可只重算已有日志） |
| `eval_variant.sh` | 对某变体最新检查点做一次论文口径评估（自动设 `PYTHONPATH`，保证与训练同代码） |
| `launch_variants.sh` | 启动 / 重启变体训练 |
| `summary.sh` | 各实验进度与奖励趋势一屏汇总 |
| `watchdog.sh` | 训练看门狗：定期巡检 + 崩溃自动续训（`--resume`）+ 定期自动评估 |
| `render_test.py` / `make_demo_preview.py` | 离屏渲染验证 / 演示帧拼图 |
| `convert_go2w.py` / `fix_go2w_usd.py` | 作者 Go2-W MJCF → Isaac Lab USD 转换与修复（多余 articulation root、缺 ContactReportAPI、prim 层级） |
| `go2w_policy_check.py` / `go2w_model_info.py` | 在 MuJoCo 里按作者部署契约跑官方权重 / 导出模型信息 |
| `inspect_checkpoints.py` | 检查点检查工具 |
| `README.md` | 训练服务器上的环境记录（镜像、容器、挂载、坑） |
| `paper_vs_impl.md` | 论文与上游实现的偏差清单（含逐项修正说明） |
| `patches/` | `upstream_dump_pickle.patch`（上游唯一必需补丁）、`paper_repro_v1.patch`（论文口径 8 项修正） |
| `variant_diffs/*.patch` | **各实验的完整实验定义**：基准 = 上游 commit `5b574775`，用 `diff -ruN` 生成（**含新增文件**，如动作噪声模块），已实测 **17/17 可 `patch -p1` 干净应用**并复现出与实验树完全一致的内容 |
| `results/*.json` | 关键评估结果：旧配置终局、`paper_v1` 中途、官方权重在 MuJoCo 的验证 |
| `evidence/` | 数字留档原件：Go2-W 模型数值（`go2w_model_facts.py/.out`）、站姿高度探针（`go2w_stance_probe.out/.json`） |

## 实验代际（`variant_diffs/` 里每个文件 = 该实验相对基准的全部代码差异）

| 代号 | 含义 | 状态 |
|---|---|---|
| `v2_exp_ori` … `v6_100hz` | 第一轮：论文口径的 5 个单点变体（姿态项形态、episode 长度、动作缩放、伺服增益、控制频率） | 已跑满 10000 iter，结论见 `docs/reference_recovery_validation.md` |
| `w1` … `w4` | 第二轮：课程权重 CW 的四种语义（`cw←cw^β` 递增 / 恒 1 / +5 s episode / 慢衰减） | `w1`、`w3` 保留跑，`w2`、`w4` 为早期淘汰 |
| `p1` … `p5` | 第三代：论文标准口径（含 Eq.3 递增 CW、Eq.4 组合、动作噪声 σ=0.02）+ 三种子 + 论文 §IV-B baseline + 门控对照 | `p1`/`p3`/`p4` 在跑 |
| `g1`、`g2` | Go2-W（论文机器人）上的论文标准口径，两个种子 | 在跑 |
| `x1`、`p2` | 被替代 / 已停（保留供追溯） | 停止 |

## 复现一份实验

`variant_diffs/*.patch` 是**完整**的实验定义：基准是**上游 `Kitjesen/recovery` 的 commit `5b574775`**，用 `diff -ruN` 生成（含新增文件），因此可直接应用到一份干净的上游检出（已实测 17/17 全部干净应用）。

```bash
# 1. 取一份干净上游
git clone --depth 1 https://github.com/Kitjesen/recovery.git recovery_test
git -C recovery_test fetch --unshallow   # 浅克隆不能推/取任意 commit，补全历史
git -C recovery_test checkout 5b57477598963695be27474c0ba941a1ba4afd38

# 2. 应用某个实验定义（-p1）
cd recovery_test && patch -p1 < /path/to/variant_diffs/<代号>.patch

# 3. 放到训练服务器的 reference_project/recovery_variants/<代号>/（容器内即 /workspace/variants/<代号>）
#    用脚本启动（脚本里已含每个变体的 GPU、seed、env 数）
./launch_variants.sh <代号>
./watchdog.sh                # 自动纳入巡检与定期评估

# 4. 论文口径评估
./eval_variant.sh <代号> 512 1
```

注意：Go2-W 的两个实验（`g1`/`g2`）在脚本里被设为 **2048 envs**（4096 下 Go2-W 的网格碰撞体会让 PhysX 初始化失败）；它们复用同一个任务 id，机器人替换发生在 `env_cfg.py`（`from thunder_recovery.config.asset_go2w import GO2W_CFG`）。

论文标准口径本身（= `p1_paper_exact` 的代码，另含两处注释更正）同时保存在分支 **`refproj/recovery`**，`default.xml` 已指向它，`repo sync` 可直接得到。
