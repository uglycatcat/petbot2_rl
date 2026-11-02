# Step ONNX / MuJoCo预览（2026-11-02）

**2026-11-02桌面反馈与显示修复：** 用户确认可完成登台运动，但一直向前不停，并有两次抬头（第一次无效），完整动作质量未通过。新增每次登台前轮仅一次真实离地的要求，下一轮训练须落实；当前第16轮继续、固定副本不改。显示修复已发布`771d7a5`，策略包仍是第14轮seed71；详见[反馈与修复](step_desktop_feedback_20261102.md)。

用户已授权在`rl_project/std_s2s_step_project`实现独立sim2sim测试程序：启动静止，Space执行最新已审查策略，R完整重置。此授权替代此前step不做sim2sim的范围限制。`rl_project/std_train_step_project`按repo建立空分支，不迁移训练；第15轮既有训练继续。

## 交付与运行

两项目均通过`default.xml`物化到各自`rlproj/<项目名>`分支；可核验版本见[正式项目清单](../rl_project/manifest.json)。主线只维护清单、索引与证据，ONNX和程序保存在独立项目。

```bash
repo sync rl_project/std_s2s_step_project rl_project/std_train_step_project
cd rl_project/std_s2s_step_project
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python run.py
```

Space只从初始等待开始，R重置机器人、估计器、一次性感知、控制目标和任务记录。默认预览在记录失败后继续展示同一策略，画面显式显示失败；`--stop-on-failure`在首个失败后暂停。15秒超时后的画面暂停不算停车。无规则抬轮或规则收尾。

运行端只需NumPy、MuJoCo3.8.1与ONNX Runtime CPU，不需要Isaac、Torch、PT或训练demo。源`robot_description/`保持只读，场景在内存加入2cm平台，边沿位于初始整机质心前30cm。详见[运行说明](../rl_project/std_s2s_step_project/README.md)。

## 策略选择与真实效果

选择时第15轮尚未完成独立评估。对近期已审查候选做实际MuJoCo闭环比较，默认第14轮seed71/update_000500，保留第13轮seed71对照。不是完整成功策略，也未获得用户桌面验收。

| 策略 | Isaac独立评估，64例 | 本次MuJoCo闭环 |
|---|---|---|
| 第13轮seed70 | 四轮登台64、规定直行49、完整0 | 约0.50秒前轮触边 |
| 第13轮seed71 | 四轮登台64、规定直行62、完整0 | 约0.82秒前轮触边 |
| 第14轮seed71（默认） | 四轮登台61、规定直行60、完整0 | 两前轮抬起、越沿、落顶，约1.04秒后轮离地失败 |

第13轮seed71仍是Isaac中进度更好的对照；第14轮seed71在此MuJoCo固定默认场景中的前轮过沿动作更完整，因此作为默认预览。有限6秒继续执行时两者均曾四轮在顶面，随后驶离平台，没有合格停车；已发生失败始终锁存，不能重新计成成功。完整来源与网络哈希见各`policies/*/policy.json`及`policies/selection.json`。

## 验证证据

发布记录在项目的[validation/review.json](../rl_project/std_s2s_step_project/validation/review.json)：

- 两份ONNX各2051输入对齐原Torch actor及归一化，最大绝对误差分别1.14e-5与3.22e-6。
- 1200步传感器估计/观测回放，包含复位和静止校正，观测最大误差1.13e-6；60个地形几何样本一致。
- 原训练48000条请求/目标记录重建完全一致。初版float64累加差异已改为float32并复核。
- MuJoCo源机器人质量、惯量、关节、执行器与几何物理参数逐项相同；只读源指纹保持`de1862d58f9989ae7eb0de5b20188e2ffc7052d2b9d5588446eef4b166b5f53f`。
- 启动100个控制周期不移动、不推理；Space只启动一次；R后两次真实52帧闭环逐项相同。运行时未导入Torch。
- 独立Xvfb软件渲染窗口真实执行Space/R/Space/R，策略确实执行、两次复位有效；直接读取MuJoCo显示帧并目视确认机器人与台阶。早期字体枚举错误及Xvfb黑色截屏方法已排除。

工程接口通过不代表跨引擎动作通过；后轮真实离地、实测轮速超调及停车缺失仍保留为缺陷。训练仍受GPU0/1、从零初始化与固定副本规则约束；本次导出和验证没有启动新训练。
