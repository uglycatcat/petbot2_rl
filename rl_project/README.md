# PetBot 正式项目入口

由根目录 `default.xml` 通过 repo 物化，代码分别保存到 `rlproj/<项目名>` 分支，main 只管理索引与清单。

| 项目 | 职责 |
|---|---|
| [train_upright_project_1104/](train_upright_project_1104/README.md) | 临时1104工程，21r5源码基线；当前4096环境×5000更新配置，单次从零训练已完成，无后续训练授权。 |
| `std_train_upright_project/` | 正式训练项目预留，继续保持空，留待以后。 |
| [std_s2s_upright_project/](std_s2s_upright_project/README.md) | 默认1104的5000更新seed102 ONNX，复用21r5控制器；旧21r5包保留。基本闭环有效，桌面与完整能力待验证。 |

在仓库根目录执行`repo sync rl_project/train_upright_project_1104 rl_project/std_s2s_upright_project`。已有本地修改先保存，不同步运行中的固定副本。demo保持当前版本作为辅助存档，旧upright runs与逐轮文档仅在服务器history；这次长训日志在1104项目logs/outputs，均不入Git，当前源码不依赖历史。

版本、已知问题与冻结边界见[1104说明](../docs/upright_1104.md)，完整要求见[需求基线](../docs/current_scope.md)，详细历史见[服务器定位](../docs/upright_history.md)。

STEP独立项目：[std_s2s_step_project](std_s2s_step_project/README.md)保持原第20轮seed71 ONNX预览，未同步当前v2，也未通过完整任务验收。本次不修改程序或策略。当前训练源码仍在`demo_project/step_rl_demo`，状态见[STEP交接](../docs/step_current.md)；`std_train_step_project/`保持空，两正式目录由各自`rlproj/`分支管理。旧预览发布证据仅按[STEP历史定位](../docs/step_history.md)追查，不作为当前训练基线。
