# PetBot 正式项目入口

由根目录 `default.xml` 通过 repo 物化，代码分别保存到 `rlproj/<项目名>` 分支，main 只管理索引与清单。

| 项目 | 职责 |
|---|---|
| `std_train_upright_project/` | 正式训练项目预留，保持空；用户亲自确认sim2sim可以通过后，再讨论从demo迁移。 |
| `std_s2s_upright_project/` | ONNX Runtime、WASD/Space/QE、方向键两轮调速及普通完整源范围。按用户要求默认新第26轮seed102/update_001000，保留第20轮可切换；60ms控制与训练映射一致。MuJoCo有限循环/交接有效，零输入落地后退12.1–17.0cm及运动后两轮零令漂移仍在，不作为全面升级；见[项目README](std_s2s_upright_project/README.md)与[发布记录](../docs/upright_restart26_sim2sim.md)。 |

在仓库根目录执行 `repo sync rl_project/std_train_upright_project rl_project/std_s2s_upright_project`。已有本地修改先提交，不对运行固定副本执行同步。现有训练源码仍在 `demo_project/upright_rl_demo/`；`demo_project/runs/` 仍保留有效产物与运行依赖。根目录 `history/` 仅用于后续归档，不进版本管理；归档前必须保留策略来源与依赖索引。

任务与验证边界见 [MuJoCo 方案](../docs/upright_sim2sim_20261027.md)。

Step独立项目：[std_s2s_step_project](std_s2s_step_project/README.md)按用户最新澄清保留第20轮seed71 ONNX供观察；训练代码回退第14轮且自动迭代暂停，不连带回退sim2sim。用户桌面观察到接近时一次抬头，但碰沿、上平台又抬头且不停，不能作完整通过。启动静止、Space/R及独立绘制保持，旧第14/13轮可显式选择。见[范围澄清与反馈](../docs/step_rollback14_20261103.md)。`std_train_step_project/`保持空，两正式项目仍由同名`rlproj/`分支管理。
