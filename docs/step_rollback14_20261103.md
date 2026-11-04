# Step暂停训练并回退第14轮（2026-11-03）

**用户已明确暂停当前训练与自动迭代，并要求恢复第14轮效果基线。未经用户明确恢复，不重启第21轮，也不发下一轮。** 用户仍希望查看最新进展，本次回退不改变该偏好。第14→20轮未解决停车且MuJoCo动作退步，不能用局部指标改善代替整体效果。

## 停止状态

北京时间2026-11-03 12:05，第21轮seed70/71分别完成148/131次更新后取消；两路`job_status.json`、`batch_status.json`及campaign `active.json`均为`cancelled_by_user`，`control.json`的enabled=false。守护器按本批run token终止对应进程组；两个STEP容器内训练/评估进程及本批控制器均已退出，没有启动终局评估或下一轮。专用容器保留，不停止其他项目或其他用户任务。

第21轮没有完成预算和独立终局，不作新策略能力结论。该轮`agent_review.json`已标记用户暂停并handled，旧完成/异常事件不得重新启用。训练检查点和日志完整保留；停止证据为`demo_project/runs/_step_rollback14_20261103/stop_verification.json`。

## 回退范围

- 实际训练目录是`demo_project/step_rl_demo`。`step_demo/`、`configs/`、`tools/`、`docker/`、`tests/`及忽略配置共110个文件逐字节恢复到第14轮固定`30fd5886271f7d70ed23a923647f824feb7c1f5f`；当前发布提交为`83a73a496b95877984c11c7d15b2499c54e3803f`。文档增加暂停和回退边界。
- 第15–21轮修改从当前可执行实现撤回。原开发tip `a983e893b8864f9efaf71fd04290a1332e562c5f`原另保存到`archive/step_pre_rollback14_20261103`；11月4日已删除该远程分支，仅在服务器`history/remote_archives_20261104/archive__step_pre_rollback14_20261103.bundle`保留，完整git历史及各轮固定副本、检查点、分析均保留。使用新提交记录回退，没有强推改写历史。
- 用户随后澄清：**回退仅针对训练实现，不回退sim2sim。** MuJoCo已恢复默认第20轮seed71供观察，发布`93247c346ce9894daaac5500bff5ac7cc3c895a0`；ONNX及运行时代码不变，独立绘制和Space/R保持。第14/13轮包仍可显式加载。此前`071c425`的第14轮默认选择已撤回。训练源码仍为第14轮（后续仅文档澄清提交`972798c`），训练和自动迭代继续暂停。
- `std_train_step_project`保持空。没有加载第14轮权重继续训练，也没有新建训练或短训。将来若恢复新训练，随机初始化要求仍有效。

## 验证与已知边界

107项容器CPU测试通过，110份实现/配置/工具/测试文件与第14轮逐字节一致；沿用冻结第14轮的既有PhysX/PPO工程证据，本次暂停期间未重新运行训练。验证见demo内`docs/rollback14_validation.json`及本机`baseline_equivalence.json`。

此前回退时验证了第14轮ONNX（现为可选对照）：`5764f94c587b5accb02478263ded4c588f83a76290b6f1d244006f2cae7dac1f`。真实物理数组及源模型指纹不变，启动静止、Space仅启动一次、R清空状态通过；两次严格运行52帧逐项复现，约1.04s发生后轮离地。当前运行时代码未改动，之前同运行时与旧第14轮300帧对照完全一致的证据保留。见[回退验证](../rl_project/std_s2s_step_project/validation/rollback14_20261103/review.json)。

用户此前确认第14轮可以登台，但不会完整停稳且存在多余抬头。恢复这一版本不等于完整任务通过。停车、每次登台前轮只离地一次，以及lower_body世界pitch尽量平缓、避免大幅后仰/低头，仍为待解决需求；当前第14轮尚未实现后两项新增训练约束。后续方案等待用户指令，不自动重新引入已撤回修改。

## 桌面同步

```bash
repo sync demo_project/step_rl_demo rl_project/std_s2s_step_project
cd rl_project/std_s2s_step_project
.venv/bin/python run.py
```

默认保留第20轮seed71供观察，Space执行、R重置；第14轮对照使用`--policy policies/round14_seed71`。主仓库两个manifest、项目索引与本地repo清单同步上述回退提交，`default.xml`继续使用既有独立项目分支。


## 后续范围澄清与桌面反馈

用户明确不回退sim2sim，继续观察昨晚的第20轮结果：接近台阶时只抬一次头，但前轮撞到边沿，登台后又抬头且不停。该桌面观察与训练/服务器诊断分开记录，不把目测抬头次数直接当作物理离地计数验收。

用户强调已知初始台阶信息和机器人IMU足以支持判断登台进度，任务必须能表达上台后的停车。当前已有IMU/编码器位置估计，训练还可使用真实轮位置/接触作为奖励与验收监督；第20轮36例真实短行后仍持续驱动，不能归因于没有“该停车”的任务信息。阶段收尾的学习设计和验证需要修正；此处记录要求与诊断，不代表已经实施新设计或恢复训练。
