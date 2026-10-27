# PetBot upright：MuJoCo 策略体验

加载已训练策略，体验四轮行驶中起立、两轮遥控、前轮落地与普通控制交接。没有用固定动作替代策略。源模型从工作区 `robot_description/` 只读加载；本项目不包含训练代码。

## 启动

在 PetBot 主仓库更新 `main`，然后物化这个独立项目：

```bash
repo sync rl_project/std_s2s_upright_project rl_project/std_train_upright_project
cd rl_project/std_s2s_upright_project
uv sync --locked --python 3.10
uv run --locked run.py
```

需要桌面显示和 OpenGL。策略在 CPU 推理，不需要 Isaac Lab 或 CUDA；MuJoCo 渲染使用本地图形环境。没有显示的服务器使用下方 `--headless`。环境和运行产物不入库。单独检出本分支时可用 `--model-dir <robot_description目录>`，模型指纹必须与策略一致。

| 操作 | 效果 |
|---|---|
| 按住 W / S | 前进 / 后退，最高 ±0.05 m/s |
| 按住 A / D | 左转 / 右转，最高 ±0.15 rad/s；可与 W/S 组合 |
| 释放方向键 | 请求零速度，控制器仍维持平衡 |
| Space | 四轮模式请求起立；起立中或两轮模式请求前轮落地；落地请求锁存到交接完成 |
| R | 显式恢复初始四轮状态 |
| 鼠标左键拖动 / 滚轮 | 旋转视角 / 缩放 |
| Esc | 退出 |

可以先按 W 行驶，再按 Space 起立；保持两轮后继续 WASD，最后按 Space 落地。起立前检查真实四轮接地及行驶状态，不强制停车。头和耳保持原位置目标，车架位置由策略输出，后轮接收左右速度目标。`--frame 0.15` 可更改对称普通行驶车架目标（导出范围 ±0.20 rad）；这不表示整个范围已通过 MuJoCo 验收。

## 当前效果与限制

默认权重 `cycle18_long_r42/model_598` 来自第十八批完整状态续训，实际累计 600 次 PPO 更新。选择依据与来源在 [selection.json](policies/selection.json)，网络/检查点 SHA、训练合同和 CUDA 导出校验在各策略 `manifest.json`。另保留 `cycle18_long_r43` 作同任务对照，可通过 `--policy policies/cycle18_long_r43` 选择。

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

每次使用新的输出目录，保留 `summary.json` 和逐步 `trace.jsonl`。`--snapshot` 额外生成真实渲染帧，需要 EGL；`--command-scale 1.25` 将脚本指令提升到键盘上限，默认脚本为 ±0.04 m/s、±0.12 rad/s。退出 2 表示流程失败或未完成，并不等于程序崩溃。基本遥控未通过但完整动作结束时退出 0，必须同时阅读 `functional`、覆盖和失败字段。

## 代码与控制契约

- `run.py`：GUI / 固定诊断场景入口。
- `petbot_s2s/controller.py`：58×5 观测、历史、外部模式请求与反复接管；`simulation.py`：MuJoCo 状态/接触力翻译与电机接口；`viewer.py`：按键按下/释放和显示。
- `petbot_s2s/core/`：固定训练源码的纯张量阶段/控制逻辑。`provenance.json` 记录来源和文件 SHA；不依赖当前 demo checkout 随后如何修改。
- `policies/`：小型 CPU TorchScript 网络及来源。控制链为 60 ms 策略、20 ms 冻结平衡反馈、2 ms 转向反馈；不是只有一个网络。车架位置/后轮速度目标保留，车架用源模型电机外部 PD 执行，物理限位、力矩限幅和源模型均不改。
- `checks/validate_port.py`：开发时对照固定源码、观测、接触力、方向和按键。例：`uv run checks/validate_port.py --source ../../demo_project/upright_rl_demo --output outputs/port_check.json`。源码更新后需使用 provenance 指定的提交；不应为让检查通过而改来源 SHA。

物理引擎、接触求解和显式/隐式伺服仍存在差异。接口一致不等于物理轨迹一致；应依据闭环实测决定后续训练修改。正式训练项目仍为空，训练代码和历史检查点留在原位置。

## 过渡修正版体验（2026-10-27）

保留原默认包。体验本次已通过GPU短训/重载和MuJoCo180秒检查的工程候选：

```bash
uv run python run.py --policy policies/correction20_gate_r42
```

这份权重从原r42继承后更新8次，仍是0.05m/s、0.15rad/s的低速工程候选，**不是全模型能力范围正式训练结果**。新控制器让起立/落地保持切换方向，并继承实际前向速度；两轮段恢复WASD。落地使用body姿态和车架角的轮轴正解，前轮到位后交还普通控制。窗口按2ms物理步推进、在渲染帧间更新姿态，策略周期仍为60/20/2ms。

已实测静止/混合起落、180秒混合控制和连续2次起落；连续第3次起立超时、带转向速度接管后提前落地仍失败，不能宣称全范围通过。新旧控制器均通过35组观测对齐；拆分物理步后的395帧轨迹逐项相同。证据见[修正验证](validation/README.md#过渡修正版2026-10-27)。

`--transition-contract linear_heading`可把新控制逻辑显式用于旧网络，仅为开发干预。新候选包自身包含新契约，无需这个参数。`--takeover-yaw 0.12 --headless`可复现初始转动场景；不要隐去其失败。网络导出和工程检查均不能替代操作者验收。

### 全范围独立调用候选

策略清单的`task_profile=full_range`启用新版调用契约：起落保持切换航向，两轮阶段恢复遥控；意外前轮回落或翻倒结束本次调用，继续普通控制，下一次起立须重新按Space。中断与正常完成分别记录，真实翻倒不标为正常四轮姿态。每次调用清理策略历史/反馈缓存，同时保留实际物理状态和连续电机目标。

历史策略包保持其原合同。新版控制端已完成MuJoCo前轮回落/翻倒的状态故障注入与源码对齐检查，接口通过不等于学习策略已能在全范围稳定工作；训练候选和实际闭环证据见主仓库`docs/upright_cycle_v21.md`。
