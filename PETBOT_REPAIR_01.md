# 2026-10-15 接口修复与风险核查

昨晚固定版本 `9553fef` 及全部运行副本保留。本次不启动完整训练扫描。

## 修复

- 明确 Lab 3 / PhysX 根姿态 xyzw，统一课程、初态分类、潜势、固定序列；机身重力/线速度/角速度从当前根姿态和世界速度计算，避开复位时未失效的派生量缓存。
- 新初态库 schema 2 明确 quaternion_order=xyzw、root_velocity_frame=world_com；旧库必须显式 `--allow-legacy-xyzw-dataset` 才能读取。检查模型、USD 指纹、关节顺序、时步与 split/seed。没有转换或覆盖旧库。
- 旧检查点默认拒绝在修复后的任务语义下评估；显式 `--allow-cross-version-checkpoint` 可做诊断，结果同时记录训练与评估版本，不冒充旧结果复现。
- 新任务 `PetBot-Recovery-Repair-v0` 独立验证有界动作：RSL-RL 原生 Beta 的采样、log-prob、KL；修正缩放熵的常数项，初始确定性均值对应物理默认构型，位置/轮速初始 std 分别 .2/.1，可依状态学习，不再施加共同硬下限。保留动作范围与模型参数。旧任务未切换分布；新任务保留历史观测、先不开课程，奖励权重不变，以分离实现修复与奖励重设计。
- 日志按所有环境统计状态相关 std，追加最小 std 和均值最大幅度；评估动作统计仅计各环境首个 episode，轨迹标记有效首回合，非有限轨迹值以 null 留痕而非丢失整份评估。

## 验证

服务器容器 `petbot-train`、GPU3 实测，原训练副本与模型不变：

| 检查 | 结果 | 证据（仓库根相对路径） |
|---|---|---|
| PhysX 已知姿态、复位、速度、10 帧历史、局部/全量自动复位 | 全部通过；新计算速度与更新后的原生机身速度一致 | `demo_project/runs/_repair01/runtime_check_v2.json` |
| 成功事件与稳定奖励正控制 | 默认站姿 64/64 成功，每回合只发一次成功事件；真实失败优先于超时 | 同上 |
| 初始策略确定性评估 | 85/512 成功，与开环成功样本一致；末态稳定 83/512（旧开环为 84/512），无动作裁剪 | `_repair01/initial_policy.json` |
| PPO 与检查点重载 | 64 环境 × 12 iteration，学习耗时 58 秒，退出 0；重载后 11/64 成功、11/64 末态稳定，无动作裁剪，仅证明链路可用 | `_repair01/ppo_smoke.log`、`checkpoint_eval.json` |
| 续训兼容性 | 新检查点继续 2 iteration（约 9 秒）退出 0；旧检查点因 state_contract 不匹配退出 1 | `_repair01/resume.log`、`resume_old_rejected.log` |
| 初始随机探索对照 | 位置/轮速 std .2/.1 与 .02/.01 均 0/512 成功和末态稳定；两组动作裁剪均 0/1,792,000 | `_repair01/stochastic_initial.json`、`stochastic_small.json` |

表内缩写 `_repair01/` 均指 `demo_project/runs/_repair01/`。集成及 PPO 使用 `4728953`，噪声对照 `eaf8d5d`，续训保护 `1683488`；后续文档提交不改变任务实现指纹。CPU 17 项测试仅辅助验证独立 MuJoCo 旋转参考与分布概率/梯度/导出契约，不作为 PhysX 验收替代。

新增真实失败优先于超时的处理，避免错误 bootstrap；旧评估入口停止使用，改用 `scripts/petbot_sweep2_eval.py` 的固定独立初态首回合评估。评估和续训均校验任务实现及物理契约；续训不允许跨状态契约。程序异常以非零退出码报告。

可复验入口（容器内项目目录）：`scripts/petbot_runtime_check.py --help`、`scripts/petbot_sweep2_eval.py --help`。短程训练参数：`scripts/train.py --headless --task PetBot-Recovery-Repair-v0 --num_envs 64 --max_iterations 12 --seed 42 --logger tensorboard agent.petbot_collection.num_states=64 agent.save_interval=1`；使用容器 `/isaac-sim/python.sh`，由 `CUDA_VISIBLE_DEVICES=3` 选择 GPU。

## 尚未解决的学习风险

有界分布不保证有效探索或恢复能力，Beta 仍可能变得过窄，必须观察执行方差。原有最终成功门槛与奖励稀疏、基于成功 EMA 的课程停滞、固定序列的 PhysX 与 MuJoCo PD 差异属于待验证的训练设计/物理迁移问题，不能用本次接口修复宣称已解决。默认构型正控制已通过，随机探索对照未通过。实测说明仅降低固定探索尺度仍不足，尚不能证明噪声是全部失败原因；下一步应拆分进入恢复域与低速保持的训练信号，验证接近恢复完成时可降低噪声，最终独立评估仍沿用严格成功判据。
