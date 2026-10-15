# PetBot 实验训练项目

本目录放我们自己的训练项目，独立于 `reference_project/` 的参考实验。根目录 [default.xml](../default.xml) 统一管理 checkout；[manifest.json](manifest.json) 记录来源、备份分支和已核对提交。main 只管理索引、清单和忽略规则；嵌套代码在本仓库 `demoproj/<项目名>` 分支版本化。

| 项目 | 底座 | 分支 | 状态 |
|---|---|---|---|
| [recovery_agile_01](recovery_agile_01/PETBOT.md) | WBC-AGILE `6830cf9`，借鉴 robot_lab 混合动作 | `demoproj/recovery_agile_01` | 第三轮六组 2048×1500 已启动（`cffc86d`），见[第三轮记录](../docs/petbot_sweep3_training.md)；第二轮 8 组全部跑满，24 次策略评估全部 0/512，见[第二轮记录](#第二轮过夜训练2026-10-14)；首轮 11 组 2048×3000 扫描 + 确定性独立评估完成：**无一组学到起身**（最好 5 组 0.225~0.227 与开环地板基线 0.213 持平）；见[首版扫描记录](recovery_agile_01/PETBOT_SWEEP_01.md) |
| [jump_rl_demo](jump_rl_demo/README.md) | 新线，不从上游派生 | `demoproj/jump_rl_demo` | 2026-10-15 已登记，`1b61916`。本机已验证前轮可离地，见[跳跃预研](../docs/jump_prestudy.md)。训练代码未写 |

## 同步与开发纪律

- 已初始化 repo 的工作区，更新主线清单后可用 `repo sync demo_project/recovery_agile_01` 物化该项目；repo 的 manifest checkout 与主仓库 main 是两份 Git 状态，主仓库 `git pull` 本身不会更新所有嵌套 checkout。
- 同一 demo 同时只由一端修改；另一端运行固定 commit 的独立副本。禁止同步正在训练的代码副本。
- 修改前检查、同步远程分支；未提交/未推送的改动先保全，分叉先合并。repo 管理检出，提交与推送仍需在 demo checkout 中执行，不能在 detached HEAD 上遗留提交。
- 每个完整修改及时 commit、push；阶段结束不得只留本机副本。首次分支推送成功后再写清单，新增/删除/改变来源或分支时同步更新本索引、manifest.json 和 default.xml。
- **我方工具/脚本**（容器定义与挂载约定、并行扫描调度、确定性评估、地板基线）备份在 `demoproj/<项目名>-tools` 分支，内容放 `tools/petbot_trial/`，不进 `default.xml`。注意 demo 仓库带 git-lfs，**基于 demo 历史开新分支推送需要上传全部 LFS 对象**，所以工具分支做成**独立孤儿分支**（只含 `tools/`）；取用方式：`git fetch ours demoproj/<项目名>-tools` 后把 `tools/petbot_trial/` 拷到 `demo_project/demo_env/`。
- 开发用完整 Git 历史；日志、检查点、生成资产与初态库不直接提交 Git，记录代码 commit、模型内容指纹、配置和产物位置。

在 repo 创建的 checkout 里，先检查 `git remote -v`；清单 remote 名为 `ours`，本机首次 clone 的 remote 名为 `origin`，不要假定两者相同。首次开发可用 `git switch -c demoproj/recovery_agile_01 --track ours/demoproj/recovery_agile_01` 建立跟踪分支；若已有同名分支，切回已有分支，不要覆盖。干净工作区内 `git pull --ff-only` 后开始修改，分叉则保全并处理。

当前桌面 checkout 是独立 clone；清单已登记，后续由 repo 按同一路径物化。完整控制契约、容器内命令与验证边界见项目 [PETBOT.md](recovery_agile_01/PETBOT.md)。2026-10-13 在训练服务器 `petbot-train` 容器内跑通资产转换、PhysX 导入、任务实例化、pre-learn 与冒烟训练（证据与两处适配见 PETBOT.md 的「服务器验证记录」）。

## 首轮结果复核与解释边界（2026-10-14）

15:03 在训练服务器核对磁盘产物：11 个变体均保存了 `model_2999.pt`，每组确定性评估为 512 episodes，结果与首轮汇总一致。`base_s42` 为 116/512（22.65625%），同批初态的开环默认构型基线为 109/512（21.28906%），零动作基线为 0/512。尚不能据此宣称已学到有效的动态起身能力。

**解释边界**：21.3% 是持续发送默认构型指令后的成功率，不能解释为“21% 初态已满足成功判据”或机器人完全不运动。扫描记录同时报告初态构型条件通过率为 0；因此仅排除初始已成功样本，未必能排除这些开环可恢复样本。奖励信号、探索噪声与初态分布是待进一步对照验证的原因，当前数据不充分排除训练预算或其它因素。嵌套 demo 的 PETBOT.md / PETBOT_SWEEP_01.md 中相关强因果表述应按此边界理解。

AGILE 同时在继续训练：日志 `reference_project/agile_env/logs/agile_standup_20261013_1742.log` 到 10774/100000，iteration time 7.47 s，日志 ETA 约 7 天 7 小时；最新检查点为 `reference_project/WBC-AGILE/logs/rsl_rl/stand_up_t1/2026-10-13_09-42-56_stand_up_t1/model_10750.pt`。这些是训练进度，不能替代独立恢复成功率评估。

### 探索标准差复核（2026-10-14）

读取 `demo_project/runs/_logs/<变体>.log` 最后一次 `Mean action std`：base_s42=0.04、base_s43=0.01、gains2x=0.03、gains_down=0.08、ppo64=0.06；decimation5=0.05、reward_config_heavy=0.08、spawn1m=0.10、reward_still=0.30、episode15=0.95、ppo_explore=188.59。这些是日志打印的动作标准差均值，不是逐通道检查点分析。最好五组的探索已明显收缩，但不是严格为零；增强探索组则出现标准差极大，与动作裁剪到 [-1,1] 的配置并存，需要核查饱和比例。后续须同时控制探索收缩与发散，不能只增加 entropy 或根据初始标准差推断训练末期状态。

## 第二轮过夜训练（2026-10-14）

代码固定为 `9553fef0a5c7c8782b9c09ab494b594e910171d1`，已推送 demo 分支。共同调整为潜势差塑形、首次连续成功奖励、回报归一化、有上下界的探索分布与渐增动作惩罚；A 无历史/课程，B 加 10 帧历史，C 再加负 pitch 倒地采样课程，每种 seed 42/43；另加 C 的探索下限 0.03 双种子，共 8 组。每组 2048 envs × 3000 iteration，从头训练。机制与边界见 [PETBOT_SWEEP_02.md](recovery_agile_01/PETBOT_SWEEP_02.md)。

实测：13 项 CPU 测试通过；完整 C 配置 64 envs × 10 iteration 冒烟通过，actor/critic 270/350 维、std≈0.35、回报归一化已启用；64 初态确定性策略评估退出码 0、成功率 0，仅用于验证链路。尚无本轮有效起身结论。

20:10 定时消息已返回会话。20:15 从具备 Docker 权限的会话启动独立调度进程；16,384 个训练初态采集完成后复用，20:31 前 8 组进程均已错峰启动，20:38 核查全部进入 PPO（各组 iteration 1–21），均通过回报归一化与各自标准差边界断言。早先 systemd user 服务调用 Docker 因 socket 权限失败，未启动训练，失败记录保留。未修改系统权限。

运行副本 `demo_project/runs/sweep2_*` 固定 commit，不同步或覆盖；只使用 `petbot-train` 的 GPU2/3，AGILE 原训练继续。调度工具已推送 `demoproj/recovery_agile_01-tools`（`52bd740`）。`demo_project/runs/_sweep2/status.json` 与各训练日志用于核查；评估计划为相同 512 个独立初态上的 default/zero/sequence 基线与 750/1500/2999 检查点，按数据 hash 配对，汇总 `SUMMARY.md` / `summary.json`，区分救回开环失败样本和丢失开环成功样本。评估自动排队，失败不会自动重训。

64 初态 PhysX 预检查：固定目标序列与保持默认构型都成功 13/64（20.3125%），成功样本完全相同（新增救回 0、丢失 0），状态张量 hash `d5ee0125586f6358e9f28c9485e85d4d847e4baa144116147f99095098ce5667`。源 checkout 的 `artifacts/sweep2_{default64,sequence64,smoke_eval}.json` 留存逐样本与轨迹，日志在 `runs/_logs/sweep2_*`。这只说明该目标序列在当前 PhysX 隐式伺服与这 64 个初态下没有额外收益，不等价于 MuJoCo 外部 PD 或实机程序，也不能否定实机恢复思路。正式 512 初态仍按计划执行。

正式 512 初态默认构型预检查已完成：85/512 成功（16.6015625%），84/512 末态稳定（16.40625%），见 `_sweep2/default_precheck.json`；状态 hash `1ee3c29a955f034c88d3d6f2f8f6f81c0443d714da6b56e086e65ec1c6e290ae`。这份库保存为 `_sweep2/eval_states.pt`，供自动评估复用。启动期实测约 24–53 s/iteration，20:38 八组均进入学习后最新约 23–26 s/iteration；按该速度估计 3000 次需要约 19–22 h，明早优先读取实际到达的 750/1500 中途检查点与评估，不宣称已跑满。自动调度已完成 default 复测（与预检查 512 条逐样本结果完全一致）并进入 zero；`summary.json`/`SUMMARY.md` 已生成，策略行需等待检查点评估。

## 第二轮终局（2026-10-15）

8 组全部完成 3000 iteration，750/1500/2999 检查点共 24 次评估全部 0/512，低于同初态默认构型基线 85/512。末次平均 std 0.15–0.54，但动作裁剪比例 23.92%–89.88%；课程成功率 EMA 无可见进展。完整结果、证据及推测边界见[终局报告](../docs/petbot_sweep2_results.md)。本节取代上面的运行中状态与耗时预测。

### 终局解释修正（2026-10-15 深入诊断）

确认 Lab 3 xyzw 与任务 wxyz 混用导致课程分组方向错误；复位后重力缓存陈旧导致旧固定序列诊断未进入翻身分支，旧序列 85/512 不能作为恢复效果。独立修正诊断为 35/512（新增救回 2、丢失 52）。24 次策略零成功结论保留，旧姿态分组与课程解释撤回；证据与改进优先级见[深入诊断](../docs/petbot_sweep2_results.md#深入诊断与改进优先级2026-10-15-1208)。该诊断时训练代码尚未修复；后续修复见下一节。

## 接口修复验收（2026-10-15）

已修复 xyzw、复位派生状态、真实失败与超时重叠、旧检查点评估/续训兼容保护；独立 Repair 任务验证有界 Beta 动作及默认构型初始化。容器内 PhysX 集成检查、64 环境 × 12 iteration PPO、检查点重载通过。确定性初始策略 85/512，与开环成功样本一致；两档随机噪声对照仍 0/512，不能据此启动完整扫描或宣称学会恢复。详见[修复记录](recovery_agile_01/PETBOT_REPAIR_01.md)。

## 第三轮正式训练（2026-10-15）

A（仅修复）/B（恢复反馈与状态相关探索）/C（再加实测初态课程）各 seed 42/43，固定 `cffc86d`，2048 环境；GPU3 四组、GPU2 两组，14:13 已全部进入 PPO。B/C 各 128×100 短训、PhysX 正控制、概率契约及检查点重载均通过；全量训练库 16,384 个状态，课程易池 2,489 个倒地状态。先跑 1500 iteration，250/750/1499 使用同一 512 独立初态评估。default 复测 85/512，尚无正式策略效果结论。代码定义见 [PETBOT_SWEEP_03.md](recovery_agile_01/PETBOT_SWEEP_03.md)，实际验收、启动与查询见[第三轮记录](../docs/petbot_sweep3_training.md)。
