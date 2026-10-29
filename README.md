# PetBot upright：MuJoCo 策略体验

通过ONNX Runtime加载已训练策略，体验四轮行驶中起立、两轮遥控、前轮落地与普通控制交接。源模型从工作区 `robot_description/` 只读加载；本项目不包含训练代码。

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
| 按住 Q / E | 普通四轮下抬头 / 低头，以0.5 rad/s调整两侧共同髋角，松键保持；策略调用期间禁用 |
| 释放方向键 | 请求零速度，控制器仍维持平衡 |
| Space | 四轮模式请求起立；起立中或两轮模式请求前轮落地；落地请求锁存到交接完成 |
| R | 显式恢复初始四轮状态 |
| 鼠标左键拖动 / 滚轮 | 旋转视角 / 缩放 |
| Esc | 退出 |

可以先按 W 行驶，再按 Space 请求起立；保持两轮后继续 WASD，最后按 Space 落地。头和耳的独立关节保持原位置目标；Q/E驱动的是左右车架/髋关节，改变头身整体俯仰。两轮时共同髋角由策略自主用于平衡。正常落地后回默认0rad，异常退出保持当时车架目标并交还普通控制，不等待回零、不复位实际姿态。

`--frame 0.15`设置初始普通车架目标；所有策略包的普通车架范围均来自源模型[-1,1.2]rad。这个范围是控制器允许范围，不表示各角度均已通过起立验证。窗口显示操作者请求、实际采用的命令、实测速度和普通车架目标。

**当前默认第23批C58仍可能在高速/较大车架角下起立失败。** 新包保留训练中的连续准备控制和8秒起立时限，允许策略在离地前减速、调整共同髋角；没有强制停车或回零脚本，宽范围鲁棒性尚未通过。普通控制完整范围与策略分开。要用与此前体验一致的手动输入幅度，可显式设置普通与两轮参数，物理能力与策略范围不变：

```bash
uv run --locked run.py --forward-speed 0.15 --turn-rate 0.5 --ordinary-forward-speed 0.15 --ordinary-turn-rate 0.5
```

## 第23批C58终局策略预览（2026-10-29）

默认`cycle23_course_r58/model_999`来自第23批课程组seed58的1000更新终局，原始检查点SHA以`bb3bbb53`开头。使用外层、完整父反馈和内部反馈三张ONNX图，保留60/20/2ms时序，并同步v25两轮轮速端点映射、连续准备与默认髋位落地。默认选择从[selection.json](policies/selection.json)读取；模型包[manifest.json](policies/cycle23_course_r58/manifest.json)记录检查点与每张网络的SHA及导出校验。

同历史协议的Isaac低速180秒完整64/64、基本63/64；0.15m/s、0.5rad/s中速60秒完整62/64、基本48/64。高速、AW/DW高段差异、极端初态角仍有失败，不将包声明范围当作可靠能力范围。

发布前只做必要短测：MuJoCo混合0.15m/s、0.5rad/s保持18秒完成起落及基本遥控；+0.15rad静止起立保持12秒完成并回默认髋位。ONNX一致性、观测/反馈时序、Q/E、普通完整范围、意外倒下退出和窗口加载检查通过。它们不是长时或全范围验收，桌面实际体验仍由用户确认，详见[验证记录](validation/README.md#第23批c58预览2026-10-29)。

原第21批B保留，可在同样参数下切换比较：

```bash
uv run --locked run.py --policy policies/cycle21_parent_b_r42 --forward-speed 0.15 --turn-rate 0.5
```

## 历史第21批终局策略预览（2026-10-28）

此前默认`cycle21_parent_b_r42/model_299`来自完整循环第21批正式PPO终局，另有`cycle21_parent_a_r43`可选。它们不是`full_range21_gate_r42`工程8更新模型，也不包含第22/23批。**尚未通过完整范围或用户sim2sim验收**。历史来源、选择依据和Isaac结果见[selection_cycle21.json](policies/selection_cycle21.json)。

B在Isaac中速60秒原评估/新种子完整60/64、62/64，优于A的49/64、55/64；B全车架范围有退步，因此A也保留。两份终局包都包含完整父反馈网络，按原60/20/2ms控制时序执行，不能只替换旧外层网络。网络CUDA1024样本输出差0，35组观测与240子步目标对齐通过；这些工程检查不等于动作质量通过。

窗口中`--forward-speed`、`--turn-rate`仅设置策略调用期间的速度/转向限幅；普通控制的幅度由独立`--ordinary-forward-speed`、`--ordinary-turn-rate`设置，默认使用源模型完整上限±0.75m/s、±15rad/s，与所选策略及其限幅无关。普通驱动不使用策略的命令加速度或轮速变化率；实际加速度仍受源执行器力矩和接触动力学约束。省略策略参数时使用对应包声明范围，该范围不等于已验证的可靠能力。组合命令按左右后轮±25rad/s联合可达域投影，速度和转向不能同时取得各自单轴最大值。

无窗口脚本的`--forward-speed`、`--turn-rate`为场景请求幅度，默认0.04m/s、0.12rad/s；允许在源模型范围内设置，策略阶段再单独按包范围处理。`--ordinary-*`仅影响窗口按键幅度。日志分别保存原始操作者请求、有效命令和实测响应。

```bash
# 第21批A候选，同样的中速输入
uv run --locked run.py --policy policies/cycle21_parent_a_r43 --forward-speed 0.15 --turn-rate 0.5
# 显式选择历史第18批策略
uv run --locked run.py --policy policies/cycle18_long_r42
```

实际MuJoCo中，B完成低速180秒完整/基本遥控；中速60秒完成但停车漂移44.46mm，基本评分未通过。较高速度、全速、第二次起立与转弯接管仍失败。真实MuJoCo场景及保留的失败见[第21批验证](validation/README.md#第21批终局预览2026-10-28)。意外前轮回落/翻倒会结束本次调用并交还普通控制；再次起立须重新按Space，不自动重起。

## 历史第18批效果与限制

此前默认权重 `cycle18_long_r42/model_598` 来自第十八批完整状态续训，实际累计 600 次 PPO 更新。当时的选择依据与来源在 [selection_cycle18.json](policies/selection_cycle18.json)，网络/检查点 SHA、训练合同和 CUDA 导出校验在各策略 `manifest.json`。另保留 `cycle18_long_r43` 作同任务对照，可通过 `--policy policies/cycle18_long_r43` 选择。

默认车架下，已在 MuJoCo 完成一条 **180 秒两轮混合遥控→前轮落地→普通控制交接**：包含前后、左右、曲线及停车，基本遥控指标通过。该结果是固定场景诊断，不是随机初态成功率，也不是实机证明。

仍有明确失败：静止指令落地后可能卡在交接，+0.15 rad 初始车架出现落地超时，连续第二次起立出现超时。窗口会暂停并显示原因，按 R 复位；不会自动重置掩盖失败。重复起落、任意航向/车架/速度和任意时刻切换还未验证成熟。测试细节与失败见 [验证记录](validation/README.md)。

![两轮支撑的实际 MuJoCo 帧](validation/20261027/two_wheels.png)

## 无窗口复现

```bash
uv run --locked run.py --headless --scenario mixed --output outputs/mixed_01
uv run --locked run.py --headless --scenario mixed --hold-seconds 180 --seconds 200 --output outputs/long_01
uv run --locked run.py --headless --scenario stationary --output outputs/stationary_01
uv run --locked run.py --headless --scenario mixed --cycles 2 --seconds 65 --output outputs/repeated_01
```

每次使用新的输出目录，保留 `summary.json` 和逐步 `trace.jsonl`。`--snapshot` 额外生成真实渲染帧，需要 EGL；可用速度参数或`--command-scale`在源模型范围内调整场景请求。退出 2 表示流程失败或未完成，并不等于程序崩溃。基本遥控未通过但完整动作结束时退出 0，必须同时阅读 `functional`、覆盖和失败字段。

## 代码与控制契约

- `run.py`：GUI / 固定诊断场景入口。
- `petbot_s2s/controller.py`：58×5 观测、历史、外部模式请求与反复接管；`simulation.py`：MuJoCo 状态/接触力翻译与电机接口；`viewer.py`：按键按下/释放和显示。
- `petbot_s2s/core/`：固定训练源码的纯张量阶段/控制逻辑。`provenance.json`记录来源和部署扩展；普通控制在`ordinary.py`独立实现，`controller.py`负责输入所有权及默认落地目标。不依赖当前demo checkout随后如何修改。
- `policies/`：CPU ONNX网络及来源。运行时只加载manifest列出的`.onnx`，不回退到`.pt`；保留TorchScript文件仅供转换与审计。外层、内部反馈及其base输出、完整父网络均已转换。控制链仍为60ms策略、20ms冻结平衡反馈、2ms转向反馈；张量状态/控制计算仍用PyTorch，神经网络推理全部由ONNX Runtime执行。物理限位、力矩限幅和源模型不变。
- `checks/export_onnx.py`：用`uv run --locked --group export python checks/export_onnx.py`重建ONNX并逐输出检查。manifest保留源文件SHA、ONNX SHA及数值容差；图转换不等于闭环验收。
- `checks/operator_controls.py`：真实MuJoCo普通满幅、Q/E、输入所有权和禁止PT运行加载检查。
- `checks/validate_port.py`：开发时对照固定源码、观测、接触力、方向和按键。例：`uv run checks/validate_port.py --source ../../demo_project/upright_rl_demo --output outputs/port_check.json`。源码更新后需使用 provenance 指定的提交；不应为让检查通过而改来源 SHA。

物理引擎、接触求解和显式/隐式伺服仍存在差异。接口一致不等于物理轨迹一致；应依据闭环实测决定后续训练修改。正式训练项目仍为空，训练代码和历史检查点留在原位置。

## 过渡修正版体验（2026-10-27）

历史工程包仍可显式选择。体验此前已通过GPU短训/重载和MuJoCo180秒检查的工程候选：

```bash
uv run python run.py --policy policies/correction20_gate_r42
```

这份权重从原r42继承后更新8次，仍是0.05m/s、0.15rad/s的低速工程候选，**不是全模型能力范围正式训练结果**。新控制器让起立/落地保持切换方向，并继承实际前向速度；两轮段恢复WASD。落地使用body姿态和车架角的轮轴正解，前轮到位后交还普通控制。窗口按2ms物理步推进、在渲染帧间更新姿态，策略周期仍为60/20/2ms。

已实测静止/混合起落、180秒混合控制和连续2次起落；连续第3次起立超时、带转向速度接管后提前落地仍失败，不能宣称全范围通过。新旧控制器均通过35组观测对齐；拆分物理步后的395帧轨迹逐项相同。证据见[修正验证](validation/README.md#过渡修正版2026-10-27)。

`--transition-contract linear_heading`可把新控制逻辑显式用于旧网络，仅为开发干预。新候选包自身包含新契约，无需这个参数。`--takeover-yaw 0.12 --headless`可复现初始转动场景；不要隐去其失败。网络导出和工程检查均不能替代操作者验收。

### 全范围独立调用候选

策略清单的`task_profile=full_range`启用新版调用契约：起落保持切换航向，两轮阶段恢复遥控；意外前轮回落或翻倒结束本次调用，继续普通控制，下一次起立须重新按Space。中断与正常完成分别记录，真实翻倒不标为正常四轮姿态。每次调用清理策略历史/反馈缓存，同时保留实际物理状态和连续电机目标。

历史策略包保持其原合同。新版控制端已完成MuJoCo前轮回落/翻倒的状态故障注入与源码对齐检查，接口通过不等于学习策略已能在全范围稳定工作；训练候选和实际闭环证据见主仓库`docs/upright_cycle_v21.md`。

历史工程候选通过`--policy policies/full_range21_gate_r42`显式选择；当前默认已改为上方第23批正式终局C58策略。无窗口诊断可用`--forward-speed`和`--turn-rate`设置幅度；取值须在该包声明范围内，控制器仍施加联合轮速限幅。`summary.json`分别记录正常完成与`interrupted_calls`，脚本不会在中断后把原定落地请求变成新起立。该候选全范围稳定性尚未通过。
