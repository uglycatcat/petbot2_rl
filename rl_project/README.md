# PetBot 正式项目入口

由根目录 `default.xml` 通过 repo 物化，代码分别保存到 `rlproj/<项目名>` 分支，main 只管理索引与清单。

| 项目 | 职责 |
|---|---|
| `std_train_upright_project/` | 正式训练项目预留，保持空；用户亲自确认sim2sim可以通过后，再讨论从demo迁移。 |
| `std_s2s_upright_project/` | ONNX Runtime推理、WASD/Space及普通Q/E抬头低头；方向键在线调整两轮线/角速度上限，Shift精调，普通完整源范围独立，正常落地回默认。当前默认新系列第20轮seed102/update_001000的独立单ONNX；旧C58与B保留供对照。用户确认基本流程和运动改善、约0.6m/s较稳；零输入交接锁存和延迟已修复，落地后仍后退12–14厘米，真正两轮微漂、双向0.75、宽初态与轮速超调仍待改善，范围/失败与启动方式见[项目README](std_s2s_upright_project/README.md)。 |

在仓库根目录执行 `repo sync rl_project/std_train_upright_project rl_project/std_s2s_upright_project`。已有本地修改先提交，不对运行固定副本执行同步。现有训练源码仍在 `demo_project/upright_rl_demo/`；`demo_project/runs/` 仍保留有效产物与运行依赖。根目录 `history/` 仅用于后续归档，不进版本管理；归档前必须保留策略来源与依赖索引。

任务与验证边界见 [MuJoCo 方案](../docs/upright_sim2sim_20261027.md)。

Step独立项目：[std_s2s_step_project](std_s2s_step_project/README.md)提供ONNX/MuJoCo预览，启动静止、Space执行、R重置；默认第14轮seed71，保留第13轮seed71对照。用户桌面确认可登台，但重复抬头与停车未解决；显示计算已与物理分开，默认目标60 FPS，实际桌面效果待复测。见[发布证据](../docs/step_sim2sim_20261102.md)和[反馈修复](../docs/step_desktop_feedback_20261102.md)。`std_train_step_project/`保持空目录预留，训练仍在step demo。两者分别由`rlproj/std_s2s_step_project`和`rlproj/std_train_step_project`版本化，执行`repo sync rl_project/std_s2s_step_project rl_project/std_train_step_project`同步。
