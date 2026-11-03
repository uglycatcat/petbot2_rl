# upright新第26轮seed102：最新体验预览

2026-11-03。用户要求报告最近完整轮次并同步策略。最新完整为第26轮，第27轮仍在GPU2/3原队列训练；本次只发布sim2sim，没有修改或重复启动训练。

## 选择与能力结论

选择第26轮seed102的独立1000更新终局，训练固定8bdd13e8c655b6bac06a7ca20b90cfe3cd348d90，run_id 0d070899-3e7a-4131-bb6b-dfae0a0313f7；检查点SHA 6a371ba96df2e04049fca7ed253ddef009375abeb05dd2c9795a8d6c61994ba2。随机初始化、无历史父网络。seed102原生训练范围±.75，低速/高范围60秒完整各32/32，宽初态29/32；seed101只到±.6且稳定性较差。

**完整流程保留，不构成全面升级。** seed102高范围基本遥控20/32；持续±.75请求实测+.412/−.465m/s，宽初态与零输入落地未全面通过。本轮整体宽初态相对第25轮退步；发布预览不改变[第26轮验收](upright_restart_round26_results.md)的拒绝能力升级结论。

## 发布内容

`rl_project/std_s2s_upright_project`固定提交`2ba0bac807374b44448aceaa7680d77f77f3fedf`、分支`rlproj/std_s2s_upright_project`。默认`policies/restart26_s102`，仅新ONNX与manifest，没有PT运行加载；主清单verified_commit、默认selection、源控制provenance和说明同步。default.xml的同名分支不变。原第20轮包和控制文件未修改，可显式切换，旧C58也保留。

新网络39×5输入、3动作，60ms策略/2ms物理；第26轮BALANCE共同轮速中心.6映射由精确复制的controller_v26.py执行，live_ordinary_command保持普通输入所有权。普通WASD/QE与全部物理范围、两轮方向键调速不变。观测方法AST与轮几何源码和第20轮一致；第20轮继续使用原控制器/30ms，避免新映射污染旧策略。未迁入critic、奖励或课程。

## 实测与失败

- ONNX数值最大误差1.43e-6；180组源码观测/目标逐项相同，旧第20轮180组回归有效；27姿态几何、普通双向约±.736m/s、Q/E四轮支持、Space历史/状态、意外退出/重新请求有效。
- MuJoCo静止保持6秒及0.15m/s、0.5rad/s混合保持18秒均完成起立、两轮、落地和普通交接。另三条42秒静止/前后运动后松键均完成并最终停车，不能据此判定动作质量通过。
- 三组首次四轮接触后仍后退17.0/15.4/12.1厘米，约.54秒才停；LAND→HANDOFF附近仍有负轮请求。两轮零输入均速约+.006/+.067/+.090m/s（后两项为运动后松键），请求与平滑参考均为0，说明残余来自策略响应，并非按键锁存。
- 持续+.15输入在普通接管后实测+.144m/s；交接中松键与合成异常退出后均归零、无自动重起。普通控制并未延长策略驱动。
- 各轨迹目标±25、髋范围、源力矩约束有效；实际轮速峰35.94rad/s超调仍在。源MJCF只读。4秒真实窗口加载通过（Xvfb软件OpenGL），不等于用户桌面验收。
- 首次导出因手动隐藏CUDA而被源隔离守护拒绝，尚未产出；保留容器原GPU2隔离环境后CPU导出成功。初次两个交接断言使用旧30ms窗口，60ms下边界仍有最大3.82e-6rad/s浮点余量；检查改为按包周期并保留1e-5边界容差，下一帧精确为0。零输入原轨迹重验、交接重跑有效，没有为检查改变策略/控制器。无训练中断。

详细记录：[预览review](../rl_project/std_s2s_upright_project/validation/20261103_restart26/review.json)，含原始轨迹哈希、完整失败与检查边界。本次没有开展新PPO、批量能力筛选或宣称sim2sim通过。

## 桌面同步与运行

在桌面主仓库更新main后执行：

```bash
repo sync rl_project/std_s2s_upright_project
cd rl_project/std_s2s_upright_project
uv run --locked run.py --forward-speed 0.15 --turn-rate 0.5
```

默认即第26轮seed102；方向键继续调两轮速度上限。对照此前认可的第20轮：

```bash
uv run --locked run.py --policy policies/restart20_s102 --forward-speed 0.15 --turn-rate 0.5
```

这次同步是供用户比较最新策略，并非推荐以第26轮替代第20轮成为能力基线。正式训练目录保持空；新第27轮现有事件迭代、GPU2/3及从零约束不变，无周期定时器或goal。
