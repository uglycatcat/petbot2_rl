# PetBot upright：MuJoCo 策略体验

通过ONNX Runtime加载已训练策略，体验四轮行驶中起立、两轮遥控、前轮落地与普通控制交接。源模型从工作区 `robot_description/` 只读加载；本项目不包含训练代码。

当前默认为**新系列21r5 seed102的1000更新ONNX**，配套该轮固定控制器。正常落地交给普通控制并响应当前输入；LAND近地限制反向轮目标、加快轮目标回零。低速和完整流程已有改善，但落地仍有少量移动，双向0.75m/s尚未达到。默认两轮幅度0.15m/s、0.5rad/s，可用方向键调整；普通四轮保持完整源范围。

## 启动

在 PetBot 主仓库更新 `main`，然后物化这个独立项目：

```bash
repo sync rl_project/std_s2s_upright_project rl_project/std_train_upright_project
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

**当前21r5 seed102仍未证明双向0.75m/s和宽髋角起立能力。** 起立沿用该轮训练的12秒时限、单网络连续控制，允许必要减速/调髋，没有强制停车或回零脚本。普通控制完整范围与策略分开。要用与此前体验一致的手动输入幅度，可显式设置普通与两轮参数，物理能力与策略范围不变：

```bash
uv run --locked run.py --forward-speed 0.15 --turn-rate 0.5 --ordinary-forward-speed 0.15 --ordinary-turn-rate 0.5
```

## 当前21r5预览（2026-11-04）

默认`restart21r5_s102`，训练提交`8cac727`、种子102、1000更新，从随机初始化训练；单ONNX、30ms策略/2ms物理，明确选择`restart_v21r5`控制器。轮目标±25rad/s、源模型和力矩保持。对应原生60秒低速完整32/32、基本31/32，高范围完整/基本30/32；两次两轮失稳保留，不能以完整流程代替能力验收。

隔离MuJoCo八场景完成；静止落地后机体后退约0.24cm、真实后轮轴总移动约1.07cm。请求±0.75时实际约+0.468/−0.520m/s，尚未通过停稳及满速要求。当前发布不含21r6的LAND髋投影修改；详见[发布验证](validation/20261104_restart21r5/review.json)。

第20轮原ONNX＋正常交接修正版保留，可直接对照：

```bash
uv run --locked run.py --policy policies/restart20_s102 --forward-speed 0.15 --turn-rate 0.5
```

第20轮交接修正证据见[历史验证](validation/20261103_round20_handoff/review.json)。原第26轮和C58仅作历史保留，不作为当前推荐。

## 无窗口复现

```bash
uv run --locked run.py --headless --scenario mixed --output outputs/mixed_01
uv run --locked run.py --headless --scenario mixed --hold-seconds 180 --seconds 200 --output outputs/long_01
uv run --locked run.py --headless --scenario stationary --output outputs/stationary_01
uv run --locked run.py --headless --scenario mixed --cycles 2 --seconds 65 --output outputs/repeated_01
```

每次使用新的输出目录，保留 `summary.json` 和逐步 `trace.jsonl`。`--snapshot` 额外生成真实渲染帧，需要 EGL；可用速度参数或`--command-scale`在源模型范围内调整场景请求。退出 2 表示流程失败或未完成，并不等于程序崩溃。基本遥控未通过但完整动作结束时退出 0，必须同时阅读 `functional`、覆盖和失败字段。

## 代码、验证与历史

`run.py`为GUI/固定场景入口；`restart.py`与`restart_core/`实现单ONNX适配和固定控制/观测，来源见`restart_provenance.json`。`simulation.py`负责源模型物理接口，`viewer.py`处理按键及显示。运行只加载包manifest列出的ONNX，不回退PT，不依赖训练checkout。当前21r5为30ms策略/2ms物理，不能只换网络而忽略包的控制时序。

策略和控制器身份见[默认包manifest](policies/restart21r5_s102/manifest.json)，工程与闭环证据见[验证索引](validation/README.md)。普通模式使用独立源范围；实际加速度受源力矩/接触约束，组合速度/转向受左右轮±25rad/s联合可达域限制。接口一致不代表跨引擎轨迹一致，最终仍需用户桌面体验。

新系列结果见主仓库[历程索引](../../docs/upright_restart_history.md)，旧第26轮已废弃，旧C58等包仅保留历史审计用途。重启前长篇实验记录已移到服务器`history/upright/`，查阅方式见[归档说明](../../docs/upright_history.md)；策略包与验证JSON保留原路径，本次没有改网络或控制代码。
