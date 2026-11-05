# PetBot upright：MuJoCo 策略体验

通过ONNX Runtime加载已训练策略，体验四轮行驶中起立、两轮遥控、前轮落地与普通控制交接。源模型从工作区 `robot_description/` 只读加载；本项目不包含训练代码。

当前默认为**1104工程独立从零训练的5000更新、seed102 ONNX**，策略包`upright1104_5000_s102`，配套原21r5控制器。正常落地交给普通控制并响应当前输入；LAND近地限制反向轮目标、加快轮目标回零。静止和混合遥控基本闭环已检查，落地质量及双向0.75m/s尚未全面验收。默认两轮幅度0.15m/s、0.5rad/s，可用方向键调整；普通四轮保持完整源范围。

## 启动

在 PetBot 主仓库更新 `main`，然后物化这个独立项目：

```bash
repo sync rl_project/std_s2s_upright_project
cd rl_project/std_s2s_upright_project
uv sync --locked --python 3.10
uv run --locked run.py --forward-speed 0.15 --turn-rate 0.5
```

需要桌面显示和 OpenGL。策略在 CPU 推理，不需要 Isaac Lab 或 CUDA；MuJoCo 渲染使用本地图形环境。没有显示的服务器使用下方 `--headless`。环境和运行产物不入库。单独检出本分支时可用 `--model-dir <robot_description目录>`，模型指纹必须与策略一致。

| 操作 | 效果 |
|---|---|
| 按住 W / S | 前进 / 后退；普通四轮默认请求 ±0.75 m/s，上述命令将策略期间限为 ±0.15 m/s |
| 按住 A / D | 左转 / 右转；普通四轮默认请求 ±15 rad/s，上述命令将策略期间限为 ±0.5 rad/s，可与 W/S 组合 |
| ↑ / ↓ | 两轮线速度上限增加 / 减少0.05 m/s；按住可连续调整，范围0至所选策略包上限 |
| → / ← | 两轮角速度上限增加 / 减少0.1 rad/s；按住可连续调整，范围0至所选策略包上限 |
| Shift + 方向键 | 精调：线速度每次0.01 m/s，角速度每次0.02 rad/s |
| 按住 Q / E | 普通四轮下抬头 / 低头，以0.5 rad/s调整两侧共同髋角，松键保持；策略调用期间禁用 |
| 释放 WASD | 请求零速度，控制器仍维持平衡；释放方向键保留当前上限 |
| Space | 四轮模式请求起立；起立中或两轮模式请求前轮落地；落地请求锁存到交接完成 |
| R | 显式恢复初始四轮状态 |
| 鼠标左键拖动 / 滚轮 | 旋转视角 / 缩放 |
| Esc | 退出 |

可以先按 W 行驶，再按 Space 请求起立；保持两轮后继续 WASD，最后按 Space 落地。头和耳的独立关节保持原位置目标；Q/E驱动的是左右车架/髋关节，改变头身整体俯仰。两轮时共同髋角由策略自主用于平衡。正常落地后回默认0rad，异常退出保持当时车架目标并交还普通控制，不等待回零、不复位实际姿态。

`--forward-speed`、`--turn-rate`设定窗口启动时的两轮上限，运行中可用方向键调整，无需重启。可在四轮时预设，或在两轮时边行驶边调；只改变策略期间的遥控限幅，普通四轮性能独立。上限松键后保持，按R复位机器人也保留，重启程序才恢复启动参数。W/S分别请求当前线速度幅度的正/负值，A/D同理；例如从0.15按一次↑后，W/S目标为±0.20 m/s，实际指令仍沿用原有平滑。上限为0时该轴请求零速；起立/落地沿用原速度继承和航向保持，不由新按键改写。组合指令仍受左右轮联合可达域限制。

`--frame 0.15`设置初始普通车架目标；所有策略包的普通车架范围均来自源模型[-1,1.2]rad。这个范围是控制器允许范围，不表示各角度均已通过起立验证。窗口显示原始操作者请求`Operator`、限幅/投影后目标`Target`、平滑及阶段处理后的实际策略指令`Command`、实测速度`Measured`和当前两轮上限。普通满幅请求可使Operator显示0.75，即使两轮Target只有0.15；判断响应应对照Command与Measured。

**当前5000更新包仍未证明持续双向0.75m/s和宽髋角起立能力。** 起立沿用该轮训练的12秒时限、单网络连续控制，允许必要减速/调髋，没有强制停车或回零脚本。普通控制完整范围与策略分开。要用与此前体验一致的手动输入幅度，可显式设置普通与两轮参数，物理能力与策略范围不变：

```bash
uv run --locked run.py --forward-speed 0.15 --turn-rate 0.5 --ordinary-forward-speed 0.15 --ordinary-turn-rate 0.5
```

## 当前5000更新发布（2026-11-05）

默认`upright1104_5000_s102`，训练提交`7ce7bc50`，4096环境、seed102、5000更新，从随机初始化训练；checkpoint为`update_005000.pt`。使用单ONNX、30ms策略/2ms物理和`restart_v21r5`控制器。当前[selection](policies/selection.json)与[包manifest](policies/upright1104_5000_s102/manifest.json)记录模型、检查点和控制契约哈希；该发布只替换默认网络包，没有修改运行控制逻辑或源模型。

1507份实际MuJoCo观测的ONNX/PT最大差异8.35e-7，180组源码观测/目标与27姿态几何对照有效。静止、混合遥控两个场景各完成起立→两轮→落地→普通控制，无中断，目标/力矩约束通过；实测轮速仍有超调。训练内最终开发探针完整/基本遥控31/32，不能替代独立能力或桌面验收。落地再抬头、零输入残余运动及满速目标仍需观察；详见[发布验证](validation/20261105_1104_5000/review.json)和[主项目结果](../../docs/upright_1104.md)。

原21r5 seed102的1000更新ONNX保留，手动对照：

```bash
uv run --locked run.py --policy policies/restart21r5_s102 --forward-speed 0.15 --turn-rate 0.5
```

原21r5的八场景、端点速度和后轮轴移动记录仅属于该旧包，见[原发布](validation/20261104_restart21r5/review.json)。第20轮交接修正版也保留在`policies/restart20_s102`，其余旧包只供按需追溯。

## 无窗口复现

```bash
uv run --locked run.py --headless --scenario mixed --forward-speed 0.15 --turn-rate 0.5 --output outputs/mixed_01
uv run --locked run.py --headless --scenario mixed --hold-seconds 180 --seconds 200 --output outputs/long_01
uv run --locked run.py --headless --scenario stationary --forward-speed 0.15 --turn-rate 0.5 --output outputs/stationary_01
uv run --locked run.py --headless --scenario mixed --cycles 2 --seconds 65 --output outputs/repeated_01
```

每次使用新的输出目录，保留 `summary.json` 和逐步 `trace.jsonl`。`--snapshot` 额外生成真实渲染帧，需要 EGL；可用速度参数或`--command-scale`在源模型范围内调整场景请求。退出 2 表示流程失败或未完成，并不等于程序崩溃。当前restart包应同时检查`completed_cycles`、`interrupted_calls`、`last_exit_code`和真实轨迹；不使用旧C58的`functional`评分。以上长时/重复命令是按需验证示例，不表示已通过这些场景。

## 代码、验证与历史

`run.py`为GUI/固定场景入口；`restart.py`与`restart_core/`实现单ONNX适配和固定控制/观测，来源见`restart_provenance.json`。`simulation.py`负责源模型物理接口，`viewer.py`处理按键及显示。运行只加载包manifest列出的ONNX，不回退PT，不依赖训练checkout。当前包为30ms策略/2ms物理，不能只换网络而忽略包的控制时序。

策略和控制器身份见[默认包manifest](policies/upright1104_5000_s102/manifest.json)，工程与闭环证据见[验证索引](validation/README.md)。普通模式使用独立源范围；实际加速度受源力矩/接触约束，组合速度/转向受左右轮±25rad/s联合可达域限制。接口一致不代表跨引擎轨迹一致，最终仍需用户桌面体验。

当前训练源码、已完成长训及能力边界见主仓库[1104说明](../../docs/upright_1104.md)。旧轮次长篇文档和runs仅在服务器`history/upright/`，按[归档说明](../../docs/upright_history.md)检索，不依赖已移走的旧文档链接。本次原始MuJoCo记录在本项目ignored的`outputs/1104_5000_20261105/`，JSON摘要与ONNX保留在Git；训练PT和TensorBoard位于1104训练项目，仅服务器保存。
