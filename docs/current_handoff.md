# 当前交接

更新：2026-11-17。新增本机姿态快照测试；upright与STEP的状态及授权保持下述边界。

## 本机手动姿态快照 demo

独立入口为 [pose_snapshot_demo](../robot_controller_demo/test/pose_snapshot_demo/README.md)。用户需求：默认位置加载机器人，手动拨动/翻倒时其余位置关节保持默认目标，记录可复用的摔倒姿态；只需旋转、关节构型和高度，不保留摆姿势产生的水平偏移。使用小UI按钮，不使用操作快捷键。面板提供保存、加载最新、选择快照、复位、暂停/继续，并显示IMU与基座高度。新保存为v2，旧v1直接兼容；加载统一水平归中、保留原四元数与关节角、高度仅向上修正到至少1mm地面间隙，然后清零速度/外力并暂停。保留`--load latest`或指定JSON；前轮被动，位置保持受原电机力矩限制。

**四份用户录制为重要数据，必须保全。** 用户已授权将test、四份原始JSON和说明提交到Git。原件位于demo的`snapshots/`，内容未改，连同`manifest.json`逐文件放行版本管理；后续新录制默认忽略。本机额外只读备份`protected_poses/20261117_first_four/`继续忽略，原件和备份均不得删除、覆盖或作测试输出。固定编号为`pose_01`至`pose_04`；其他程序使用独立无窗口`pose_io.PoseLoader`，可直接恢复到调用方的MuJoCo model/data。具体代码、调用契约与可运行例子见[加载说明](../robot_controller_demo/test/pose_snapshot_demo/README.md#其他程序直接加载姿态)。

2026-11-17 开发验证：10项MuJoCo状态/文件/公共接口测试通过，包括四份JSON恢复到调用方已有model/data；暂存源码的干净副本在无DISPLAY、无本机备份目录时，四个固定ID和跨工作目录脚本入口均可加载。四份重要旧姿态均完成加载及UI检查，旋转/关节角不变、x=y=0，分别上移约1.006/1.054/1.499/1.003mm，最小地面间隙均1mm；原件和备份校验一致。新版保存测试只写临时目录。Linux有DISPLAY时使用进程内X11/XWayland，退出请求排队并等待viewer线程结束。新加载方式的人工体验待用户确认；模型未改，未训练、未操作服务器。

## upright：1104长训与当前默认策略

当前完整训练工程为[train_upright_project_1104](../rl_project/train_upright_project_1104/README.md)，源码基线为21r5 seed102。用户单独授权的GPU3、4096环境×5000更新已完成，训练提交`7ce7bc50`，随机初始化，无显存中断或恢复。默认MuJoCo包已发布为`upright1104_5000_s102`，取自`update_005000.pt`；原21r5包保留。demo仍为`a821990`辅助存档，其21r6代码不是当前训练源码；正式std_train目录仍为空。

5000更新开发验证32/32进入两轮、完整/基本遥控31/32；1507份真实观测ONNX/PT最大差异8.35e-7，静止和混合MuJoCo各完成一循环。两轮跟踪有改善，约2000更新后收益放缓；新包尚无桌面验收，落地多余抬头/残余移动、双向连续±0.75m/s及宽初态不能据此认定通过。细节与证据集中在[1104说明](upright_1104.md)。

**单次训练与发布已结束，没有运行队列，不启动下一轮或自动迭代。** `project.json`仍保留该次`training_enabled=true`，不是新训练授权；工程/评估及自动迭代开关为false。仅同次受迫中断可恢复，已完成运行不得续训；GPU2/3、源模型只读、无周期定时器或goal保持。

本次日志/检查点在1104项目的`logs/gpu3_4096x5000_s102_20261104/`与`outputs/gpu3_4096x5000_s102_20261104/`，均不进Git；MuJoCo原始轨迹在sim2sim的`outputs/1104_5000_20261105/`。旧轮次仅在服务器history，按[历史定位](upright_history.md)查找。当前身份见各项目README、release/selection及[主清单](../rl_project/manifest.json)；源码不依赖历史文件。

## 独立STEP v2

**用户2026-11-05已明确授权从零训练与持续自动迭代：4096×128×1000、GPU2/3每卡两路，并录制批次最佳策略MP4。新批为step_v2_round03_20261105，实际状态看control/active与日志。** 当前代码、两批结果、状态路径和待解决事项只维护在[STEP当前交接](step_current.md)，用户目标与4096环境/128步采样/1000更新预算见[STEP需求](step_scope.md)。

STEP与upright独立，只参考项目管理方式，不继承算法结构。旧STEP轮次和旧说明已移入服务器history，不默认加载；需追查时见[STEP历史定位](step_history.md)。当前扩容配置与工程证据见STEP交接；源机器人和sim2sim保持不变。
