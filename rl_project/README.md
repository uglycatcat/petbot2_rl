# PetBot 正式项目入口

由根目录 `default.xml` 通过 repo 物化，代码分别保存到 `rlproj/<项目名>` 分支，main 只管理索引与清单。

| 项目 | 职责 |
|---|---|
| `std_train_upright_project/` | 正式训练项目预留，保持空；用户亲自确认sim2sim可以通过后，再讨论从demo迁移。 |
| `std_s2s_upright_project/` | ONNX Runtime、WASD/Space/QE、方向键两轮调速及普通完整源范围。按用户要求默认21r5 seed102及对应训练控制器，20保留对照；停车与满速未过。见[项目README](std_s2s_upright_project/README.md)与[发布记录](../docs/upright_restart21r5_sim2sim.md)。 |

在仓库根目录执行 `repo sync rl_project/std_train_upright_project rl_project/std_s2s_upright_project`。已有本地修改先提交，不对运行固定副本执行同步。现有训练源码仍在 `demo_project/upright_rl_demo/`；`demo_project/runs/` 仍保留有效产物与运行依赖。旧upright系列已归档至服务器 `history/upright/`，不进版本管理；新系列、当前基准和在用依赖保留原路径，见[归档说明](../docs/upright_history.md)。

任务与验证边界见[需求基线](../docs/current_scope.md)与[当前MuJoCo发布](../docs/upright_restart21r5_sim2sim.md)。

Step独立项目：[std_s2s_step_project](std_s2s_step_project/README.md)按用户最新澄清保留第20轮seed71 ONNX供观察；训练代码回退第14轮且自动迭代暂停，不连带回退sim2sim。用户桌面观察到接近时一次抬头，但碰沿、上平台又抬头且不停，不能作完整通过。启动静止、Space/R及独立绘制保持，旧第14/13轮可显式选择。见[范围澄清与反馈](../docs/step_rollback14_20261103.md)。`std_train_step_project/`保持空，两正式项目仍由同名`rlproj/`分支管理。
