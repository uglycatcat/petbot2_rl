# Step暂停训练并回退第14轮（2026-11-03）

**用户已明确暂停当前训练与自动迭代，并要求恢复第14轮效果基线。未经用户明确恢复，不重启第21轮，也不发下一轮。** 用户仍希望查看最新进展，本次回退不改变该偏好。第14→20轮未解决停车且MuJoCo动作退步，不能用局部指标改善代替整体效果。

## 停止状态

北京时间2026-11-03 12:05，第21轮seed70/71分别完成148/131次更新后取消；两路`job_status.json`、`batch_status.json`及campaign `active.json`均为`cancelled_by_user`，`control.json`的enabled=false。守护器按本批run token终止对应进程组；两个STEP容器内训练/评估进程及本批控制器均已退出，没有启动终局评估或下一轮。专用容器保留，不停止其他项目或其他用户任务。

第21轮没有完成预算和独立终局，不作新策略能力结论。该轮`agent_review.json`已标记用户暂停并handled，旧完成/异常事件不得重新启用。训练检查点和日志完整保留；停止证据为`demo_project/runs/_step_rollback14_20261103/stop_verification.json`。

## 回退范围

- 实际训练目录是`demo_project/step_rl_demo`。`step_demo/`、`configs/`、`tools/`、`docker/`、`tests/`及忽略配置共110个文件逐字节恢复到第14轮固定`30fd5886271f7d70ed23a923647f824feb7c1f5f`；当前发布提交为`83a73a496b95877984c11c7d15b2499c54e3803f`。文档增加暂停和回退边界。
- 第15–21轮修改从当前可执行实现撤回。原开发tip `a983e893b8864f9efaf71fd04290a1332e562c5f`另保存到`archive/step_pre_rollback14_20261103`，完整git历史及各轮固定副本、检查点、分析均保留。使用新提交记录回退，没有强推改写历史。
- MuJoCo项目`rl_project/std_s2s_step_project`默认恢复第14轮seed71，发布`071c42504a8de9ed1c3e474aed758c68149fc51f`。原ONNX字节与哈希保持，独立绘制、Space执行和R重置保持；第20/13轮包仍可显式加载。
- `std_train_step_project`保持空。没有加载第14轮权重继续训练，也没有新建训练或短训。将来若恢复新训练，随机初始化要求仍有效。

## 验证与已知边界

107项容器CPU测试通过，110份实现/配置/工具/测试文件与第14轮逐字节一致；沿用冻结第14轮的既有PhysX/PPO工程证据，本次暂停期间未重新运行训练。验证见demo内`docs/rollback14_validation.json`及本机`baseline_equivalence.json`。

MuJoCo默认加载第14轮ONNX：`5764f94c587b5accb02478263ded4c588f83a76290b6f1d244006f2cae7dac1f`。真实物理数组及源模型指纹不变，启动静止、Space仅启动一次、R清空状态通过；两次严格运行52帧逐项复现，约1.04s发生后轮离地。当前运行时代码未改动，之前同运行时与旧第14轮300帧对照完全一致的证据保留。见[回退验证](../rl_project/std_s2s_step_project/validation/rollback14_20261103/review.json)。

用户此前确认第14轮可以登台，但不会完整停稳且存在多余抬头。恢复这一版本不等于完整任务通过。停车、每次登台前轮只离地一次，以及lower_body世界pitch尽量平缓、避免大幅后仰/低头，仍为待解决需求；当前第14轮尚未实现后两项新增训练约束。后续方案等待用户指令，不自动重新引入已撤回修改。

## 桌面同步

```bash
repo sync demo_project/step_rl_demo rl_project/std_s2s_step_project
cd rl_project/std_s2s_step_project
.venv/bin/python run.py
```

默认即第14轮seed71，Space执行、R重置。主仓库两个manifest、项目索引与本地repo清单同步上述回退提交，`default.xml`继续使用既有独立项目分支。
