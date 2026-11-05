# 当前交接

更新：2026-11-05。upright已冻结；step状态与授权单独见下方，不因upright收尾改变。

## upright：1104长训与当前默认策略

当前完整训练工程为[train_upright_project_1104](../rl_project/train_upright_project_1104/README.md)，源码基线为21r5 seed102。用户单独授权的GPU3、4096环境×5000更新已完成，训练提交`7ce7bc50`，随机初始化，无显存中断或恢复。默认MuJoCo包已发布为`upright1104_5000_s102`，取自`update_005000.pt`；原21r5包保留。demo仍为`a821990`辅助存档，其21r6代码不是当前训练源码；正式std_train目录仍为空。

5000更新开发验证32/32进入两轮、完整/基本遥控31/32；1507份真实观测ONNX/PT最大差异8.35e-7，静止和混合MuJoCo各完成一循环。两轮跟踪有改善，约2000更新后收益放缓；新包尚无桌面验收，落地多余抬头/残余移动、双向连续±0.75m/s及宽初态不能据此认定通过。细节与证据集中在[1104说明](upright_1104.md)。

**单次训练与发布已结束，没有运行队列，不启动下一轮或自动迭代。** `project.json`仍保留该次`training_enabled=true`，不是新训练授权；工程/评估及自动迭代开关为false。仅同次受迫中断可恢复，已完成运行不得续训；GPU2/3、源模型只读、无周期定时器或goal保持。

本次日志/检查点在1104项目的`logs/gpu3_4096x5000_s102_20261104/`与`outputs/gpu3_4096x5000_s102_20261104/`，均不进Git；MuJoCo原始轨迹在sim2sim的`outputs/1104_5000_20261105/`。旧轮次仅在服务器history，按[历史定位](upright_history.md)查找。当前身份见各项目README、release/selection及[主清单](../rl_project/manifest.json)；源码不依赖历史文件。

## 独立STEP v2

**当前暂停，先同步问题；训练、自动迭代和工程实验须明确重新授权。** 当前代码、两批结果、状态路径和待解决事项只维护在[STEP当前交接](step_current.md)，用户目标与至少2048环境/1000更新预算见[STEP需求](step_scope.md)。

STEP与upright独立，只参考项目管理方式，不继承算法结构。旧STEP轮次和旧说明已移入服务器history，不默认加载；需追查时见[STEP历史定位](step_history.md)。当前算法、配置、固定v2产物和sim2sim未因整理改变。
