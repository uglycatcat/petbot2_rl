# PetBot 正式项目入口

由根目录 `default.xml` 通过 repo 物化，代码分别保存到 `rlproj/<项目名>` 分支，main 只管理索引与清单。

| 项目 | 职责 |
|---|---|
| [train_upright_project_1104/](train_upright_project_1104/README.md) | 临时1104工程，完整保存21r5 seed102从零训练源码/配置，当前冻结；以后从此了解有效版本。 |
| `std_train_upright_project/` | 正式训练项目预留，继续保持空，留待以后。 |
| [std_s2s_upright_project/](std_s2s_upright_project/README.md) | 保持21r5 seed102 ONNX和对应控制器；基本流程可用，落地质量及满速目标未通过。 |

在仓库根目录执行`repo sync rl_project/train_upright_project_1104 rl_project/std_s2s_upright_project`。已有本地修改先保存，不同步运行中的固定副本。demo保持当前版本作为辅助存档，全部upright历史runs与逐轮文档仅在服务器history；当前源码不依赖历史。

版本、已知问题与冻结边界见[1104说明](../docs/upright_1104.md)，完整要求见[需求基线](../docs/current_scope.md)，详细历史见[服务器定位](../docs/upright_history.md)。

Step独立项目：[std_s2s_step_project](std_s2s_step_project/README.md)按用户最新澄清保留第20轮seed71 ONNX供观察；训练代码回退第14轮且自动迭代暂停，不连带回退sim2sim。用户桌面观察到接近时一次抬头，但碰沿、上平台又抬头且不停，不能作完整通过。启动静止、Space/R及独立绘制保持，旧第14/13轮可显式选择。见[范围澄清与反馈](../docs/step_rollback14_20261103.md)。`std_train_step_project/`保持空，两正式项目仍由同名`rlproj/`分支管理。
