# 文档索引

本目录是项目的**细节文档**（"肋骨"）目录。[AGENTS.md](../AGENTS.md) 里的**进度报告是主线**——先看进度，再按需翻这里的某一篇。

所有文档面向"下一个 agent / 新会话"：只写结论与必要证据，不留对话痕迹。

## 当前文档

| 文档 | 作用 |
|---|---|
| [current_scope.md](current_scope.md) | 需求基线：目标、动作接口、边界与待定项 |
| [engineering_handoff.md](engineering_handoff.md) | 工程实施：接口契约、恢复任务设计、阶段验收 |
| [reference_recovery_validation.md](reference_recovery_validation.md) | 参考项目 recovery 的**完整记录**：结论、判据、实验矩阵、产物与查询、坑、参数速查 |

## 仓库根目录的相关文档

| 文档 | 作用 |
|---|---|
| [AGENTS.md](../AGENTS.md) | agent 入口：设备判断、全局规则、**进度报告（主线）** |
| [README.md](../README.md) | 人类入口：目录与职责、文档地图、如何配置 |

## 历史资料

[reports/](reports/) 是 2026-09-24 的网页调研报告与当时的模型快照，只用于模型与文献追溯；其中的参数与结论不随当前模型更新，也不代表当前方案。

## 维护约定

- 结构是**鱼骨**：进度报告（主线，在 AGENTS.md）→ 按需下钻到这里的某一篇细节文档。
- 需求写进 `current_scope.md`；工程实现与待定参数写进 `engineering_handoff.md`；参考项目的实测与结论写进对应的参考项目文档。
- 关键节点与结果**先写入进度报告**（带日期，一行一条），细节再展开到对应文档。
- 文档内使用相对路径；不写主机地址、账号、密码与凭据（本仓库公开）。
