# 文档索引

本目录是项目的**细节文档**（"肋骨"）目录。[AGENTS.md](../AGENTS.md) 里的**进度报告是主线**——先看进度，再按需翻这里的某一篇。

所有文档面向"下一个 agent / 新会话"：只写结论与必要证据，不留对话痕迹。

## 当前文档

| 文档 | 作用 |
|---|---|
| [upright_moving_v4.md](upright_moving_v4.md) | 运动第四轮：独立转向反馈（heading_feedback）与姿态反馈对照 |
| [upright_moving_automation.md](upright_moving_automation.md) | 运动任务的自动迭代流程：轮次登记、判定规则、gate 与分层归因 |
| [upright_moving_v5.md](upright_moving_v5.md) | 第五轮：持续朝向信用 + 纯转向占比，转向与支撑的主要增益 |
| [upright_moving_v6.md](upright_moving_v6.md) | 第六轮：学习率 1e-4 / 3e-5 对照，峰值位置与退化 |
| [upright_moving_v7.md](upright_moving_v7.md) | 第七轮：中间学习率 5e-5，天花板 61/128 |
| [upright_moving_v8.md](upright_moving_v8.md) | 第八轮：从峰值续训 + 后轮接触中断计费 |
| [upright_moving_v9.md](upright_moving_v9.md) | 第九轮：窗口进度奖励，合计 68/128 为最好 |
| [upright_moving_measurement.md](upright_moving_measurement.md) | 支撑判据的测量口径：单帧 vs 控制步窗口，以及修正后的数字 |
| [current_scope.md](current_scope.md) | 需求基线：目标、动作接口、边界与待定项 |
| [engineering_handoff.md](engineering_handoff.md) | 工程实施：接口契约、恢复任务设计、阶段验收 |
| [fixed_pose_recovery.md](fixed_pose_recovery.md) | 本机 MuJoCo 固定姿态恢复动作组，以及被动沉降姿态的采集与结果 |
| [jump_prestudy.md](jump_prestudy.md) | 前轮离地与后轮起跳的测量，以及放弃完整跳跃的决定 |
| [upright.md](upright.md) | 起立任务：前轮离地、后轮稳定支撑，允许过程位移但终局回到原位置 |
| [upright_v2_design.md](upright_v2_design.md) | jump_demo服务器复现、首版数据诊断和第二版分阶段训练设计（待实现） |
| [reference_selection.md](reference_selection.md) | 参考项目选择建议、源码证据、接入成本与待验证步骤 |
| [reference_recovery_validation.md](reference_recovery_validation.md) | 已收官参考项目 recovery 的**完整记录**：结论（含关闭决定）、判据 A/B/C/C2/D、实验矩阵、产物与查询、坑、参数速查 |

## 仓库根目录的相关文档

| 文档 | 作用 |
|---|---|
| [AGENTS.md](../AGENTS.md) | agent 入口：设备判断、全局规则、**进度报告（主线）** |
| [README.md](../README.md) | 人类入口：目录与职责、文档地图、如何配置 |

## 历史资料

[reports/](reports/) 是 2026-09-24 的网页调研报告与当时的模型快照，只用于模型与文献追溯；其中的参数与结论不随当前模型更新，也不代表当前方案。

## 维护约定

- agent 在交流、排查和实施中主动识别会影响后续决策、执行或交接的信息，及时更新文档，无需用户提醒或逐次授权；具体范围与授权边界见 [AGENTS.md](../AGENTS.md#全局规则硬约束)。
- 结构是**鱼骨**：进度报告（主线，在 AGENTS.md）→ 按需下钻到这里的某一篇细节文档。
- 需求写进 `current_scope.md`；工程实现与待定参数写进 `engineering_handoff.md`；参考项目的实测与结论写进对应的参考项目文档。
- 关键进展、需求与决策变化、实测结果、重要失败经验和阻塞项按日期写入进度报告（一行一条）；长期有效的背景、约束和决策依据同步到对应主题文档，并修正过时表述。
- 仅保留结论与必要证据，区分用户确认、实测与待验证推测；不记录对话流水，也不为无新增信息的重复检查追加记录。
- 文档内使用相对路径；不写主机地址、账号、密码与凭据（本仓库公开）。

- [起立第二版训练记录](upright_v2_training.md)：实际实现、短训失败边界、固定版本部署与评估安排。

- [10月16日起立夜间迭代](tonight_training_20261016.md)：四卡批次、三次续接、迭代依据与最终停止边界。

- [起立第三版结果](upright_v3_training.md)：四卡对照，离地回位成功与持续保持缺口。
- [起立第四版训练](upright_v4_training.md)：连续姿态稳定奖励对照与实际部署状态。

- [起立第五版训练](upright_v5_training.md)：有界探索下限对照，夜间最后一轮。

- [运动中接管首版](upright_moving_v1.md)：三路控制、真实运动初态、遥控轨迹与工程验收。
- [10月19日夜间迭代](tonight_training_20261019.md)：当前运动任务四卡训练、评估和02:00续接。

- [运动首轮终局与第二轮](upright_moving_v2.md)：16次起立失败评估、轮速限制反事实、第二轮对照与验收。

- [运动第三轮](upright_moving_v3.md)：累计位置/朝向偏差诊断、姿态反馈奖励与显式初始化对照。
