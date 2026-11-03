# PetBot 正式项目入口

由根目录 `default.xml` 通过 repo 物化，代码分别保存到 `rlproj/<项目名>` 分支，main 只管理索引与清单。

| 项目 | 职责 |
|---|---|
| `std_train_upright_project/` | 正式训练项目预留，保持空；用户亲自确认sim2sim可以通过后，再讨论从demo迁移。 |
| `std_s2s_upright_project/` | ONNX Runtime、WASD/Space/QE、方向键两轮调速及普通完整源范围。按用户要求默认新第26轮seed102/update_001000，保留第20轮可切换；60ms控制与训练映射一致。MuJoCo有限循环/交接有效，零输入落地后退12.1–17.0cm及运动后两轮零令漂移仍在，不作为全面升级；见[项目README](std_s2s_upright_project/README.md)与[发布记录](../docs/upright_restart26_sim2sim.md)。 |

在仓库根目录执行 `repo sync rl_project/std_train_upright_project rl_project/std_s2s_upright_project`。已有本地修改先提交，不对运行固定副本执行同步。现有训练源码仍在 `demo_project/upright_rl_demo/`；`demo_project/runs/` 仍保留有效产物与运行依赖。根目录 `history/` 仅用于后续归档，不进版本管理；归档前必须保留策略来源与依赖索引。

任务与验证边界见 [MuJoCo 方案](../docs/upright_sim2sim_20261027.md)。

Step独立项目：[std_s2s_step_project](std_s2s_step_project/README.md)提供ONNX/MuJoCo预览，启动静止、Space执行、R重置；默认第20轮seed71，保留第14/13轮对照。29维传感记忆、轮映射和单次离地监测已同步并验证，MuJoCo0.56s碰沿、15s仍不停，不能视为动作通过。见[发布与实测](../docs/step_round20_sim2sim.md)。窗口保持独立绘制、目标60FPS，实际性能取决于设备。`std_train_step_project/`保持空目录，训练仍在step demo；两正式项目由同名`rlproj/`分支版本化，执行`repo sync rl_project/std_s2s_step_project rl_project/std_train_step_project`同步。
