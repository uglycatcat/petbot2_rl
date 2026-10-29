# PetBot 正式项目入口

由根目录 `default.xml` 通过 repo 物化，代码分别保存到 `rlproj/<项目名>` 分支，main 只管理索引与清单。

| 项目 | 职责 |
|---|---|
| `std_train_upright_project/` | 正式训练项目预留，保持空；用户亲自确认sim2sim可以通过后，再讨论从demo迁移。 |
| `std_s2s_upright_project/` | ONNX Runtime推理、WASD/Space及普通Q/E抬头低头；普通完整源范围与策略独立，正常落地回默认。当前默认第23批C58/model_999，保留第21批B等策略对照；已同步v25连续准备、8秒起立时限和轮速映射，高速/极端角度仍不可靠，范围/失败与启动方式见[项目README](std_s2s_upright_project/README.md)。 |

在仓库根目录执行 `repo sync rl_project/std_train_upright_project rl_project/std_s2s_upright_project`。已有本地修改先提交，不对运行固定副本执行同步。现有训练源码仍在 `demo_project/upright_rl_demo/`；`demo_project/runs/` 仍保留有效产物与运行依赖。根目录 `history/` 仅用于后续归档，不进版本管理；归档前必须保留策略来源与依赖索引。

任务与验证边界见 [MuJoCo 方案](../docs/upright_sim2sim_20261027.md)。
