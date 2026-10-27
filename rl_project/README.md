# PetBot 正式项目入口

由根目录 `default.xml` 通过 repo 物化，代码分别保存到 `rlproj/<项目名>` 分支，main 只管理索引与清单。

| 项目 | 职责 |
|---|---|
| `std_train_upright_project/` | 正式训练项目预留，当前保持空；验证成熟后再迁移。 |
| `std_s2s_upright_project/` | 已训练策略的 MuJoCo 推理、键盘遥控与迁移验证。 |

在仓库根目录执行 `repo sync rl_project/std_train_upright_project rl_project/std_s2s_upright_project`。已有本地修改先提交，不对运行固定副本执行同步。现有训练源码仍在 `demo_project/upright_rl_demo/`；`demo_project/runs/` 仍保留有效产物与运行依赖。根目录 `history/` 仅用于后续归档，不进版本管理；归档前必须保留策略来源与依赖索引。

任务与验证边界见 [MuJoCo 方案](../docs/upright_sim2sim_20261027.md)。
