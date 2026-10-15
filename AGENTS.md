# PetBot 自恢复 RL —— Agent 接手说明

本文件是任何 agent / 新会话进入本项目的**唯一入口**。读完它即可知道：项目是什么、现在到哪一步、有哪些硬规则、该去读哪份文档。

## 项目目标与当前阶段

- 目标：PetBot（轮足机器狗）在平地上被推倒、滚翻或跌落后，用强化学习策略恢复到默认站立构型并保持稳定。
- 主线：Isaac Lab + RSL-RL + PPO 训练 → 策略导出 → MuJoCo 做 sim2sim 验证。
- **已有能力与训练目的**：本机 MuJoCo 固定姿态恢复已实现。用户于 2026-10-14 确认当前动作组效果可用、按是否摔倒和 pitch 符号分支的思路明确，但不能覆盖所有摔倒情况；没有全姿态成功率统计。RL 用于学习更自然、连贯、平滑的恢复动作。默认轮足构型无需单独训练站立策略；策略采用 **5 路位置 + 2 路速度**混合输出；训练**不依赖参考动作流或示范轨迹数据**，这套固定姿态恢复不是训练数据。详见[固定姿态恢复](docs/fixed_pose_recovery.md)与[需求基线](docs/current_scope.md#已有能力与强化学习的目的)。
- **当前阶段：PetBot 第三轮训练与 AGILE 参考试训并行。** recovery 线已关闭，robot_lab 已就绪未开训。AGILE 标准 `StandUp-T1-v0` 训练于 2026-10-13 启动（4096 envs / 100k iteration），仍保留运行。PetBot 前两轮未学出有效恢复，第二轮 24 次策略评估全为 0/512；状态接口修复后，第三轮 A/B/C × seed 42/43 已于 2026-10-15 14:03 启动，14:13 六组全部进入 PPO，2048 环境、暂跑 1500 iteration，尚无正式策略效果结论。见[第三轮记录](docs/petbot_sweep3_training.md)。本机跳跃预研已关闭：前轮能离地，后轮离地不到 1 mm，完整跳跃放弃。留下的是起立，见[起立](docs/upright.md)。`jump_rl_demo` 登记保留，训练改在 `upright_rl_demo`，代码未写。
- **PetBot 与参考项目的本质区别**：PetBot **本来就能轻松站住**（默认站姿是四轮着地的静稳定姿态，站立不需要策略去学）；而 `recovery` 这类轮足翻倒起身项目，研究对象默认**连"站住"都还没解决** —— 它们的站立姿态是要学出来的，所以才有高度 / 姿态塑形、ED-CW 类课程，以及"站住了但还在晃"这种失败模式。**参考项目的奖励权重、课程与成功判据不能直接照搬。**

## 第一步：判断自己在哪台设备上

本项目只在两台设备上运行，动手前先自识别：

| 设备 | 判断方法 | 角色 |
|---|---|---|
| **训练服务器** | `nvidia-smi -L` 有 4 块 RTX 4090；`$HOME` 在 `/data/...` | 全部训练、实验、渲染与评估 |
| **本机（桌面）** | `nvidia-smi -L` 只有 1 块 RTX 2080 Ti；`$HOME` 在 `/home/...` | **不做训练**；只在最后用 MuJoCo 等检验训练效果 |

两台设备通过 git 共享同一份仓库。本仓库是**公开仓库**：主机地址、账号、密钥、代理配置一律不进版本管理。

## 先读什么

文档是**鱼骨结构**：本文件的[进度报告](#进度报告文档主线)是**主线**，具体细节按需去对应的"肋骨"文档里查。

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | 本文件的[进度报告](#进度报告文档主线) | **主线**：关键节点与结果的带日期记录 |
| 2 | [README.md](README.md) | 目录与文件职责、文档地图、如何配置 |
| 3 | [docs/reference_recovery_validation.md](docs/reference_recovery_validation.md) | 已收官参考项目 recovery 的**完整记录**：结论（含关闭决定）、判据、实验、产物与查询、坑、参数速查 |
| 4 | [docs/current_scope.md](docs/current_scope.md) | 需求基线：目标、动作接口、范围与待定项 |
| 5 | [docs/engineering_handoff.md](docs/engineering_handoff.md) | 工程实施：接口契约、恢复任务设计、阶段验收 |
| 6 | [reference_project/README.md](reference_project/README.md) | 上游快照索引、来源记录与使用边界 |
| 7 | [docs/fixed_pose_recovery.md](docs/fixed_pose_recovery.md) | 本机 MuJoCo 固定姿态恢复动作组，以及被动沉降姿态的采集与结果 |

## 全局规则（硬约束）

1. **分支**：日常开发与项目文档主要在 `main` 上维护，以 `origin/main` 为共享主线。因任务隔离、实验或保存参考项目修改等需要，可以创建和使用用途明确的命名分支，无需仅因使用其他分支另行请示；参考项目修改沿用 `refproj/<项目名>` 约定。切换或合并前检查工作区，保留已有改动，避免覆盖他人工作。
2. **共享服务器纪律**：服务器上有其他用户。除全局配置与 `/tmp` 这类不可避免的变动外，**不要在家目录下新增或修改其他目录**；不碰别人的家目录与容器；不修改系统配置、不重启 docker、不使用 `sudo`（需要特权操作时先问用户）。
3. **训练只在容器内跑**，不在宿主机直接跑 Isaac Sim / 训练；宿主机只做编排、检查与看门狗。
4. **`robot_description/` 只读**：MJCF、STL 及该目录内其他文件禁止修改，模型与物理参数一律不动；只有该目录 `README.md` 中**对机器人的文字描述**允许更新。
5. **本机不做训练**：桌面机只用于最后的效果验证（MuJoCo 等）。
6. **参考项目不进版本管理**：`reference_project/*/` 是上游快照与训练产物，被 `.gitignore` 排除；项目文档、配置与 PetBot 自有代码正常提交。
7. **不拿旧设备记录当基线**：桌面机上的历史运行记录（RTX 2080Ti 等）与训练无关，不作为任何结论依据。
8. **路径**：程序用相对路径加载模型与资源，不硬编码宿主机或容器绝对路径。
9. **文档写法**：根目录 `README.md` 面向人类，主要用于部署与使用，不同步实验进度；进度维护在本文件与对应实验文档。面向"下一个 agent"，只写结论与必要证据，不留对话痕迹；过时内容直接改掉或标明日期，不与新结论并列陈述。
10. **不确定就问**：规则之间若出现冲突，或需求含义不明，先问用户再动手。
11. **主动沉淀重要信息与进度**：agent 必须在交流、排查和实施过程中自行判断新增信息是否影响后续决策、执行或交接；凡涉及目标与背景、需求或约束变化、已确认的决策及依据、关键进展、实测结果、重要失败经验或阻塞项，都应及时同步到对应文档，不等待用户提醒，也不以完成代码修改为前提。关键节点按日期写入本文件的[进度报告](#进度报告文档主线)，长期有效的信息更新到对应主题文档；同步修正受影响的旧表述。只保留结论与必要证据，区分用户确认、实测和待验证推测，不记录对话流水或无实质变化的重复检查。
12. **参考项目由 `default.xml` 清单管理**：`reference_project/` 下的项目通过 `repo` 物化。新增、删除或改变某个项目的 pin，必须在同一次改动中同步更新 `default.xml` 与 `reference_project/manifest.json`，保持清单与目录一致。
13. **参考项目的本地修改不能丢**：在服务器上跑 demo 时若改了某个参考项目的代码，要把**该项目**提交到本仓库的一个专用分支（命名 `refproj/<项目名>`，如 `refproj/recovery`），并把 `default.xml` 里该项目的 `remote`/`revision` 指向那条分支；在完成这一步之前，不要删除或覆盖那个 checkout。**为某个参考项目写的工具、脚本与实验定义**同样推到该项目的 `refproj/` 分支（如 `refproj/recovery-tools`，内容在 `tools/recovery_trial/`）；这类分支不写进 `default.xml`（它不是上游项目）。参考项目由 `repo sync` 按 `default.xml` 物化，**不要为取用参考项目或工具分支而切换主仓库工作区**；这不限制第 1 条允许的任务分支。

14. **PetBot demo 独立管理**：`demo_project/<项目名>/` 由同一 `default.xml` 通过 repo 物化，代码在 `demoproj/<项目名>` 分支版本化，main 不重复跟踪嵌套 checkout。修改前同步、完整修改后及时提交并推送；双端同一 demo 单端写入，训练固定 commit。清单、demo 索引与 manifest 同步维护；细则见[工程实施说明](docs/engineering_handoff.md#版本管理与双端协作)。

## 沟通与交付

- 只要结论 + 最小必要证据，不要过程叙述，不要重复验证已经验证过的东西。
- 只读排查默认允许；上述常规文档同步，以及已授权任务所需的命名分支创建与使用，视为持续授权，无需逐次请示。其他有副作用的变更（写文件、动服务器、推送）须在用户已授权的范围内进行，超出范围时先获得授权；文档维护不自动授权提交、推送或服务器操作。
- 完成后逐项核对：列出本次实际改动、影响范围与耗时，并区分「实测」与「推测」。
- 最小改动优先，不要为了"完整"引入用户没要的设计。

## 进度报告（文档主线）

**这是文档的骨架**：只记关键节点与结果，一行一条，细节链接到对应的"肋骨"文档。**AI 必须主动维护它**：出现影响项目推进的重要进展、需求或决策变化时及时按日期记录，并在本次工作结束前检查是否已同步；无需等待用户提出文档更新要求。长期有效的背景、约束与决策依据写入对应主题文档，不能只留在对话或本表中。

| 日期 | 事件 | 结果 |
|---|---|---|
| 2026-09-24 | 模型观察与文献调研归档 | `docs/reports/`（历史资料） |
| 2026-09-29 | 7 个上游参考项目快照入库 | [manifest.json](reference_project/manifest.json) |
| 2026-09-30 | 追加 rl_sar、recovery（累计 9 个），整理独立仓库与需求文档 | rl_sar 已构建并跑通 MuJoCo 通路 |
| 2026-09-30 | 桌面机（RTX 2080Ti）早期部分运行 512 envs × 400 iter | **已做，但无参考价值**：该机不承担训练，记录仅存档 |
| 2026-10-09 | 训练服务器部署 recovery 容器，发起论文原配置训练 | 4096 envs × 10000 iteration |
| 2026-10-10 | recovery 上游原配置训练跑满：**完全失败**（成功率 0.00%） | 见 [recovery 试训记录](docs/reference_recovery_validation.md) |
| 2026-10-10 | 按论文口径改 8 项偏差后启动 6 个并行实验（1 个论文口径基线 + 5 个变体） | 已于 2026-10-11 跑满，结果见 [recovery 试训记录](docs/reference_recovery_validation.md) |
| 2026-10-10 | 追加 Recovery_go2w 到清单；仓库远程迁至 `petbot2_rl`，文档重整为鱼骨结构 | 本文件与 [README.md](README.md) |
| 2026-10-11 | 第一轮 6 个实验跑满 10000 iteration：论文口径 C 全部 0.00%；但论文口径的 `paper_v1` 确实学会从倒地起身（倒置初态 46.8%、侧躺 38.2%），同时"站不静"（末态腿关节速度 p50 7.63 rad/s，论文要求 < 0.1） | [recovery 试训记录](docs/reference_recovery_validation.md) |
| 2026-10-11 | 定位根因（对照论文原文）：上游把课程权重 CW 实现成**衰减** `0.3×0.968^i`（第 1000 iteration 即 1e−14，行为惩罚全被关掉），论文 Eq.3 是**递增** `cw←cw^β`（0.3 → 1）；episode 应为 **5 s @ 50 Hz**；行为惩罚组按 Eq.4 只乘 CW、不含 DS | 同上 |
| 2026-10-11 | **接入论文平台**：论文用的 Go2-W / KYON 模型没有随上游发布；已从作者部署仓库取 Go2-W 模型转进 Isaac Lab（修 3 处 USD 转换问题），启动论文标准训练 | 同上 |
| 2026-10-11 | 用作者官方权重 `model_7999.pt` 在 MuJoCo（作者 MJCF + 作者推理契约）验证：**确实能自行翻身站稳**（倾角 p50 11.3°、末段最大关节速度 p50 0.010 rad/s），平台自适应判据成功率 37.5% —— 作为本机训练的**复现基准**；同时澄清"高度 > 0.42 m"是 KYON 口径，Go2-W 默认姿态实测站姿高度仅 **0.299 m**，套用必然 0% | 同上 |
| 2026-10-12 | 第三代论文标准口径实验运行中（Thunder 3 个 + 经验延续 1 个 + **Go2-W 2 个**）；Go2-W @2000 站姿达标 25.2%、末态高度 p50 0.407 m 已超官方权重（0.323 m），同口径 Thunder 仅 10.2%、关节速度 10.6 rad/s —— 瓶颈指向**被训平台**（49.3 kg + 硬伺服）而非论文方法 | 同上 |
| 2026-10-12 | 仓库治理：改过的 recovery 推为分支 **`refproj/recovery`**（`default.xml` 的 recovery 项已改指向它）；我方试训工具、补丁与全部实验定义推为分支 **`refproj/recovery-tools`**（`tools/recovery_trial/`）；文档按最新结果同步更新 | 本文件、[README.md](README.md)、[参考项目索引](reference_project/README.md) |
| 2026-10-12 | 文档自检：逐条对照磁盘/远程修正 6 处不符（Go2-W 站姿实测 0.299 m、单 run 4.4~4.9 核、开销按日志重算、参考项目物化状态、`env.sh` 子命令清单、Go2-W 资产路径）；补 3 组数字留档（模型数值 / 站姿探针 / 权重网络结构） | [recovery 试训记录](docs/reference_recovery_validation.md) |
| 2026-10-12 | 训练服务器上验证 **repo 清单链路**：`repo init -b main --repo-url <TUNA> -m default.xml` + `repo manifest` 解析通过（10 项，recovery→`ours`/`refproj/recovery`）；同时发现并记录两个环境参数（repo 工具本体需国内镜像源、`-b main` 必带） | [README.md](README.md) 的「如何配置」 |
| 2026-10-12 | 工具与状态澄清：`summary.sh` 的实验清单与 ANSI 解析修正（原来监控的是上一批已结束实验）；**看门狗按用户决定不常驻启用**，文档同步为"无自动续训 / 无定期自动评估"，并补 2 条坑（ANSI 转义、看门狗被静默杀死） | [recovery 试训记录](docs/reference_recovery_validation.md) |
| 2026-10-12 | 明确已有恢复能力与 RL 训练目的 | 用户已验证固定姿态转换可恢复站立；训练要求无需独立站立策略、位置 / 速度混合输出、无参考动作数据，目标为更自然连贯的恢复动作；见[需求基线](docs/current_scope.md#已有能力与强化学习的目的) |
| 2026-10-12 | 更新分支与文档维护规则 | 以 `main` 为主，必要时可使用命名分支；agent 主动识别并沉淀重要信息与进展，常规文档同步无需另行请示 |
| 2026-10-12 | 核查并明确参考项目选型标准 | 以恢复任务接入代码成本为主，环境更换不作主要负担；源码建议优先 AGILE / T1，robot_lab 用于混合动作参考，尚未实测或定案，见[选择依据](docs/reference_selection.md) |
| 2026-10-12 | 核查 AGILE 成熟度与容器入口 | 有官方 Dockerfile 与完整任务入口，但依赖 Lab 3 beta；公开策略清单未列 T1 起身权重，服务器启动与训练效果待实测；见[选择记录](docs/reference_selection.md#agile-的成熟度与容器运行边界) |
| 2026-10-12 | 合并桌面与服务器 main 进展 | 保留服务器 8 个新增提交及本地需求、分支规则和选型记录；同步作者权重已验证状态，AGILE 仍为待实测候选 |
| 2026-10-13 | 第三代实验终局评估（Go2-W 两个跑满 10000，Thunder 4 个取最新检查点）：**三种判据（含逐关节口径）全部 0.00%**，卡点定位为"站定"——逐关节偏差条件最多已有 57.1% 通过，但速度条件（<0.1 rad/s 保持 1 秒）通过率≈0；**发现种子方差达 8 倍**（同配置 7.4% vs 63.7%）；**发现"站得高"与"站回默认构型"冲突**（推得最高的 w3/g1 偏差通过率反而最低 1.9%/2.0%），指向目标高度设定（Go2-W 站姿 0.299 m vs 奖励目标 0.426 m） | [recovery 试训记录](docs/reference_recovery_validation.md) |
| 2026-10-13 | **关闭 recovery 试训线**（用户决定）：三代 15+ 实验未达论文判据，判断项目本身价值有限、停止投入；终局数据、判据口径修正与结论已全部同步进文档与工具分支（`refproj/recovery-tools`）；容器 `recovery-train` / `recovery-vis` **停用未删除**（检查点保留，可 `--resume` 续）；下一步另选参考项目试训 | [recovery 试训记录](docs/reference_recovery_validation.md#收官结论2026-10-13用户决定) |
| 2026-10-13 | 文档自检（收官后）：对照 git 与磁盘修正 3 处过时表述 —— 工具分支改动"尚未推送"（实为已推送，tip `06f8fd7`）、`p3`/`w3` 的 C2/D "待补"（实为已完成、6 实验三种口径全 0.00%）、种子方差"6 倍 / 11.5% vs 65.6%"（终局为 8 倍 / 7.4% vs 63.7%） | [recovery 试训记录](docs/reference_recovery_validation.md) |
| 2026-10-13 | **建好两个参考项目的专用容器（彼此独立、不混用）**：`agile-train`（Isaac Lab **3.0.0-beta2** 基础镜像，16 GB 经镜像站多连接拉取＋短超时补块补齐；装 rsl-rl-lib 5.4.1 并打上游补丁 → `StandUp-T1-v0` 环境创建成功，action_dim=23 = 全关节位置 + `lift`）与 `robotlab-train`（复用本机 `isaac-lab:2.3.2`；Go2W 环境创建成功，action_dim=16 = 腿位置 + 轮速）；容器脚本在 `reference_project/{agile_env,robot_lab_env}/`，副本已推 `refproj/agile-tools`、`refproj/robot_lab-tools` | [参考项目索引](reference_project/README.md#其它参考项目的容器环境2026-10-13-建好) |
| 2026-10-13 | **参考项目在服务器全部物化**：首次实际执行 `repo sync`（`-j4`，排除已存在的 recovery/Recovery_go2w），8 个待物化项目全部到位，10 个 checkout 的 HEAD 与 `default.xml` pin 逐一核对一致；同步记录 WBC-AGILE 的预训练权重与视频只是 LFS 指针（T1 资产走 Nucleus + `agile-download-assets`）、服务器上的 rl_sar 无 submodule/构建产物 | [参考项目索引](reference_project/README.md) |
| 2026-10-13 | 服务器建好 AGILE / robot_lab / PetBot demo 的专用容器（见下几行）；**AGILE 标准起身训练已启动**：`StandUp-T1-v0`、4096 envs、100k iteration、seed 42、tensorboard；实测 ≈7.4 s/iteration（≈8.8 天），倒地库 pre-learn 完成（65536 状态，缓存 16 MB） | AGILE 底座 + robot_lab 混合动作；全方向随机落地沉降后控制；首版训练与 TensorBoard；`demo_project/` + `demoproj/` 分支由 repo 管理并严格双端同步，见[实施约定](docs/engineering_handoff.md#petbot-demo-管理与首版交付2026-10-13-用户确认) |
| 2026-10-13 | 本地实现首个 PetBot demo `recovery_agile_01`，代码推送 `demoproj/recovery_agile_01`（`b009c06`），登记 repo 清单与 demo 索引 | 9 项 CPU 测试、静态检查与语法检查通过；源模型未改，未连接服务器/运行训练；USD/PhysX 和学习效果待验证，见 [demo 说明](demo_project/recovery_agile_01/PETBOT.md) |
| 2026-10-13 | **服务器验证 PetBot demo**（容器 `petbot-train`）：资产转换、PhysX 导入、任务实例化、pre-learn（默认站姿探针 + 倒地初态采集）、冒烟训练全部跑通；修两处适配后推送 `demoproj/recovery_agile_01`（`5622024`、`c5bc149`）；同步 demo 索引与清单 | 默认站姿实测 0.08427 m、初态库 16384/16384（接受率约 80%）；64 envs 2.7 s/iter、1024 envs 4.0 s/iter（≈8000 steps/s、显存 3.7 GB ⇒ 10000 iteration ≈ 11 h）；两处适配：容器内 git 需复刻宿主相对层级、导入关节名被加 `_1` 后缀；**未跑满 iteration、未评估、未导出**，见 [PETBOT.md](demo_project/recovery_agile_01/PETBOT.md#服务器验证记录2026-10-13) |
| 2026-10-13 | 用户决定首版训练口径：**2048 envs**（1024 太少），并**并行多组参数同训**，用于一次性比较伺服增益、奖励权重与种子方差 | 见[首版训练与扫描](demo_project/recovery_agile_01/PETBOT.md#首版训练与扫描2026-10-1314) |
| 2026-10-13 | PetBot 首版并行扫描**已部署**：11 组变体各跑固定 commit 的独立副本（`demo_project/runs/`），2048 envs × 3000 iter，**GPU3 6 组 + GPU2 3 组**并行、2 组排队（10-14 01:00/02:10 补位，08:38 全部跑满）；调度与探活脚本 `demo_project/demo_env/{launch_variants.sh,runs_probe.sh}` | 实测 Hydra 只接受 `agent.*` 覆盖，env 字段（奖励 / decimation / 回合长度）必须做代码副本；首个初态库被 6 组复用（缓存命中）；9 组并行约 7~8 s/iteration |
| 2026-10-14 | **首版扫描 + 确定性独立评估完成，结论：没有一组学到起身**。最好 5 组 0.225~0.227，而开环"恒定保持默认构型"地板基线就有 **0.213**（恒定零动作 0.000）；`reward_config_heavy` 0.084、`spawn1m` 0.053、其余 4 组 0.000（且大量 episode 因越界/非法状态提前结束） | 三条机制：① 训练期 `Recovery/success_rate` 测的是探索噪声下的策略，不能比较变体，必须用确定性评估；② 开环保持默认构型可达 21.3%，不等于初态已满足成功判据；需区分开环可恢复样本与需要学习动态恢复的样本；③ 伺服增益 0.5×~2× 不是瓶颈，控制率/回合长度/加重静止惩罚/更大噪声都更差。设计与现象见[首版扫描记录](demo_project/recovery_agile_01/PETBOT_SWEEP_01.md) |
| 2026-10-14 | 服务器工具**版本化**：新增孤儿分支 `demoproj/recovery_agile_01-tools`（内容 `tools/petbot_trial/`，含容器挂载约定、并行调度、确定性评估、地板基线与调试脚本，脚本改为向上查找 `default.xml` 定位仓库根） | 双端可复现本轮扫描；demo 仓库带 git-lfs，基于 demo 历史开分支推送需上传全部 LFS 对象，故工具分支独立成孤儿分支（`git fetch ours demoproj/recovery_agile_01-tools` 后取 `tools/petbot_trial/`，其 `INDEX.md` 是工具入口） |
| 2026-10-14 | 15:03 服务器复核：PetBot 11 组均有 `model_2999.pt`，评估 JSON 与首轮汇总一致；AGILE 日志到 10774/100000，最新检查点 `model_10750.pt`，仍在推进，尚无独立效果评估 | 修正入口过时状态；21.3% 是开环控制后的成功率，不是初态成功率，见 [复核边界](demo_project/README.md#首轮结果复核与解释边界2026-10-14) |
| 2026-10-14 | 明确文档职责与下一步验证安排 | 根 README 用于人类部署/使用，不同步进度；文档使用正式项目名。先在 `robot_controller_demo/` 实现传统固定姿态恢复，再结合构型、接触与恢复过程推进 RL 优化 |
| 2026-10-14 | **被动沉降姿态统计**（本机 MuJoCo，执行器关闭）：1000 套停稳，55 次超时后补抽。身体 +Z 与世界 +Z 的夹角最大 **121.7°**，130° 以上为 0；左侧躺、右侧躺、倒立探针都离开初始朝向 | 主簇在 110°–120°（584 套）。均匀朝向、贴地 1 cm 放下。采集与分档见 [固定姿态恢复](docs/fixed_pose_recovery.md#被动沉降) |
| 2026-10-14 | **固定姿态恢复动作组**（本机 MuJoCo，`pose_recover_demo`）：用户试过，效果可用，不能覆盖所有摔倒；没有全姿态成功率 | 未摔倒 1 s 回初始角。摔倒且 pitch < 0 为 4 s。摔倒且 pitch ≥ 0 为 8 s：车架先走 4 s，再接上负 pitch 那套。动作表见 [固定姿态恢复](docs/fixed_pose_recovery.md) |
| 2026-10-14 | 用户澄清实机经验与 MuJoCo 演示边界 | 实机更全面程序可按“车架调整后倒并使高摩擦后轮触地 → 轮驱动 → 急停翻身”恢复其测试过的所有场景；当前演示缺少部分转换，非实机版本，其覆盖不足不代表该思路无效。实机测试明细未归档，见[恢复思路](docs/fixed_pose_recovery.md#恢复思路与实机经验的边界2026-10-14-用户补充) |
| 2026-10-14 | 核对首轮最终探索标准差日志 | 最好五组 `Mean action std` 已降至 0.01–0.08；增强探索组达到 188.59。存在收缩与发散两种现象，不能仅看初始标准差或一律提高 entropy；见[探索复核](demo_project/README.md#探索标准差复核2026-10-14) |

| 2026-10-14 | 第二轮 PetBot 训练代码、测试与过夜调度已部署 | 固定 `9553fef`，13 项 CPU 测试、C 配置 64×10 冒烟与 64 初态独立评估通过；20:38 核查 8 组 2048×3000 全部进入 PPO；独立 512 初态默认基线 85/512，尚无训练效果结论；见[第二轮记录](demo_project/README.md#第二轮过夜训练2026-10-14) |
| 2026-10-15 | 第二轮 8 组训练与 27 个评估任务全部完成 | 750/1500/2999 检查点共 24 次策略评估全为 0/512，同初态默认基线 85/512；动作饱和明显，未学出恢复能力，见[终局报告](docs/petbot_sweep2_results.md) |
| 2026-10-15 | 深入诊断确认四元数顺序与复位重力缓存错误 | 旧负 pitch 课程池实际为正 pitch；旧固定序列未执行翻身，修正诊断 35/512（新增救回 2、丢失 52）；接口修复随后完成（见下一行），见[诊断报告](docs/petbot_sweep2_results.md) |
| 2026-10-15 | 修复姿态/复位接口、动作初始化及评估兼容检查 | PhysX 集成验收通过，64 环境 × 12 iteration PPO 与重载通过；默认均值复现 85/512，但两档随机探索均 0/512，训练反馈仍需优化，见[修复验收](demo_project/recovery_agile_01/PETBOT_REPAIR_01.md) |
| 2026-10-15 | 第三轮 A/B/C 双种子训练通过 GPU 门槛并启动 | 初版终态噪声默认站姿正控制 25/64 未通过；减小终态噪声后 64/64，倒置新增奖励为零。B/C 各 128×100 短训及重载通过，正式六组已启动（2048×1500），见[第三轮记录](docs/petbot_sweep3_training.md) |
| 2026-10-15 | 本机验证前轮可离地，并另开跳跃 demo 目录 | 对称轮速与车架力矩的一次约束动力学通过；`jump_demo` 时间积分里前轮离地、后轮着地并停住。四轮依次离地再落地的小跳尚未测到。用户决定据此另开跳跃训练线，训练代码未写，见[跳跃预研](docs/jump_prestudy.md) |
| 2026-10-15 | 跳跃线进入 repo 管理 | `demoproj/jump_rl_demo` 已推送（`1b61916`），`default.xml` 检出到 `demo_project/jump_rl_demo`。训练代码仍未写 |
| 2026-10-15 | 明确跳跃训练任务目标 | 独立于恢复：左右对称的前进跨栏，顺序为前轮离地、后轮离地、前轮落地、后轮落地。本阶段从静止默认姿态完成一次跳跃并回到默认姿态、零速度，位置可变。训练代码仍未写，见[跳跃预研](docs/jump_prestudy.md) |
| 2026-10-15 | 跳跃可引用做法归档 | 四足伸腿跳跃不作为依据。可引用后轮驱动自行车 UMV 的位置/速度动作和接触区分；腾空段取抛体物理。训练代码仍未写，见[跳跃预研](docs/jump_prestudy.md) |
| 2026-10-15 | jump_demo 增加后轮起跳和跨越 | 倒立不收敛位置。停稳后 Rear Hop 与 Cross 互斥可用，四轮再着地会重做后轮支撑。见[跳跃预研](docs/jump_prestudy.md) |
| 2026-10-15 | 放弃完整跳跃，另开起立 | 后轮离地不到 1 mm。`jump_rl_demo` 保留但不训练。起立见[起立](docs/upright.md)。`demoproj/upright_rl_demo`（`1d812bf`）已推送，训练代码未写 |

细节写进对应文档，本表只保留一行结论。

## 当前实现状态

- **第三轮运行中（2026-10-15）**：物理、概率、短训及重载门槛通过；固定 `cffc86d`，六组 2048×1500 全部进入 PPO，250/750/1499 配对评估已安排。最新状态以[第三轮记录](docs/petbot_sweep3_training.md)的实测状态为准。
- **起立（2026-10-15，本机）**：完整跳跃已放弃，依据见[跳跃预研](docs/jump_prestudy.md)。任务是从默认姿态维持只有后轮着地，位置最好不变，左右对称，脖子和耳朵不动。`jump_demo` 能做到，但太僵硬，只作参照。`demo_project/upright_rl_demo` 已登记，训练代码未写。见[起立](docs/upright.md)。

- **第二轮训练已结束（2026-10-15）**：8 组全部完成 3000 iteration，24 次策略评估全为 0/512。标准差有界但动作饱和明显，课程成功率 EMA 无可见进展；尚无有效恢复策略。已确认 xyzw/wxyz 混用和复位重力缓存错误，接口修复及容器内验收已完成（详见下方修复记录）；旧课程/固定序列解释失效；后续第三轮状态见上。[结果与证据](docs/petbot_sweep2_results.md)。

- **固定姿态恢复（2026-10-14，用户试操）**：`robot_controller_demo/pose_recover_demo/` 已实现。未摔倒用 1 s 回到初始角；摔倒后按 pitch 符号走 4 s 或 8 s 动作组。用户确认效果可用、思路明确，但不能覆盖所有摔倒情况。没有全姿态成功率统计。动作、判据和被动沉降数字见 [固定姿态恢复](docs/fixed_pose_recovery.md)。该动作组是构型与接触的参照，不是训练示范数据。
- **被动静止姿态（2026-10-14 实测）**：执行器关闭、均匀朝向贴地放下后，1000 套停稳，倾斜角最大 121.7°，130° 以上为 0。侧躺与倒立初态都会离开初始朝向。原始数据在 `robot_controller_demo/test/fall_pose_statistics/output/`。

- **参考项目与 PetBot 状态**：recovery 已关闭；AGILE 容器内 `StandUp-T1-v0` 标准训练**运行中**（2026-10-13 启动，4096 envs / 100k iteration，检查点与 TensorBoard 见上方进度行），robot_lab 容器已就绪未开训。PetBot `recovery_agile_01` 服务器链路已跑通，首轮 11 组扫描与确定性评估已完成——**没有一组学到从倒地起身**（最好 5 组与开环地板基线持平），详见[首版扫描记录](demo_project/recovery_agile_01/PETBOT_SWEEP_01.md)与[实施约定](docs/engineering_handoff.md#petbot-demo-管理与首版交付2026-10-13-用户确认)。
- **已有**：机器人模型（MJCF + 10 个 STL）、参考项目快照、训练服务器上的容器与训练 / 评估 / 渲染链路、recovery 的完整训练与评估结果。
- **实机恢复经验（2026-10-14 用户确认）**：更全面的实机程序可按后轮触地、轮驱动及急停翻身思路恢复其测试过的所有场景；当前 MuJoCo 程序是简化参照，非实机版本。实机测试明细尚未归档。
- **已有恢复能力（2026-10-14 用户试操）**：本机 MuJoCo 固定姿态动作组有可用恢复效果，分支按是否摔倒和 pitch 符号选择；用户确认不能覆盖所有摔倒情况。尚无该动作组的全姿态成功率，也尚无 RL 自然恢复动作的评估结果。见 [固定姿态恢复](docs/fixed_pose_recovery.md)。
- **参考分支**：`refproj/recovery`（上游 recovery + 我们的论文口径补丁，`default.xml` 已指向）、`refproj/recovery-tools`（给 recovery 试训用的脚本、补丁与全部实验定义，见该分支 `tools/recovery_trial/INDEX.md`）、`refproj/agile-tools` 与 `refproj/robot_lab-tools`（两个参考项目容器环境的版本化副本，见各自 `tools/agile_trial/`、`tools/robot_lab_trial/`）、`demoproj/recovery_agile_01-tools`（PetBot demo 的服务器工具：容器挂载约定、并行扫描调度、确定性评估与地板基线，见该分支 `tools/petbot_trial/INDEX.md`；**独立孤儿分支**，只含 `tools/`，因为 demo 仓库带 git-lfs，基于 demo 历史开分支推送要上传全部 LFS 对象）。
- **PetBot 首轮扫描已完成、训练效果未出现**：`demo_project/recovery_agile_01/` 的资产适配、混合伺服配置、随机落地初态库、PPO 任务、TensorBoard 指标与独立评估入口；2026-10-13/14 跑完 11 组（2048 envs × 3000 iter）并做确定性独立评估——最好 5 组与开环"保持默认构型"地板基线持平，**没有一组学到从倒地起身**，见 [首版扫描记录](demo_project/recovery_agile_01/PETBOT_SWEEP_01.md)。尚无跑满 10000 iteration 的对照、无策略导出、无 MuJoCo 策略推理验证。
- PetBot 动作接口（已定）：**5 个关节位置目标（`head` / `left_ear` / `right_ear` / `left_frame` / `right_frame`）+ 2 个后轮速度目标（`left_rear_wheel` / `right_rear_wheel`）**，由仿真内部伺服闭环执行；左右前轮被动；头、耳、车架与身体均可触地支撑。

## 目录职责

| 路径 | 职责 | 能否修改 |
|---|---|---|
| `robot_description/` | 机器人模型来源（MJCF + STL 网格） | 模型、物理参数与网格**只读**；`README.md` 的文字描述可更新 |
| `docs/` | 项目文档：需求、工程、参考项目进度 | 可改 |
| `reference_project/` | 上游快照与训练产物 | 不受主线版本管理；但**不放 PetBot 自有的唯一副本** |
| `demo_project/` | PetBot 自有实验项目，由 repo 物化 | 嵌套代码在 `demoproj/` 分支版本化，清单与索引在 main |
| `robot_controller_demo/` | 本机 MuJoCo 固定姿态恢复、跳跃前轮离地演示与被动沉降统计；早期键盘驾驶只作历史记录 | 可按该任务阅读、参考；源模型只读约束不变 |
| `AGENTS.md` / `README.md` | agent 入口 / 人类入口 | 可改 |

## 检查与交接

- 服务器验收以容器内 PhysX 集成检查、短程 PPO、检查点重载为准；CPU 测试只辅助验证数学与概率契约，不能替代真实仿真。
- demo 的本地测试：在 `demo_project/recovery_agile_01/` 运行 `python -m pytest -q --confcutdir=petbot_tests petbot_tests`（CPU Torch / MuJoCo 等依赖见 PETBOT.md）；它不启动 Isaac Sim 或训练。改 `pose_recover_demo/` 时运行 `python -m unittest discover -s robot_controller_demo/pose_recover_demo -q`。改 `jump_demo/` 或后轮离地可行性测试时，用 `mujoco_env` 运行 `python -m unittest robot_controller_demo.jump_demo.test_maneuver robot_controller_demo.test.rear_wheel_inverted_pendulum.test_lift_feasibility robot_controller_demo.test.rear_wheel_liftoff.test_frame_pitch_liftoff -q`。`driver_demo/` 的旧测试不必运行。
- 每次工作结束前：核对重要信息与进展是否已同步至本文件「进度报告」及对应主题文档；实现状态发生变化时同步更新「当前实现状态」。明确区分用户确认、实测与未验证内容；无新增重要信息时无需为留痕而改文档。
- `docs/reports/` 是 2026-09-24 的历史调研资料，只作模型与文献追溯，不代表当前方案。
