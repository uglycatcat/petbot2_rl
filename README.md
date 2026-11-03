# Step：ONNX → MuJoCo 预览

启动后机器人以默认姿态静止显示。按 **Space** 执行策略，按 **R** 重置机器人、策略输入、惯性估计、目标和任务记录。鼠标左键旋转视角，右键平移，滚轮缩放，Esc退出。

2026-11-02显示修复：物理/策略计算与窗口绘制分开，窗口默认目标60 FPS（可用`--render-fps`调整），画面显示实际FPS和仿真速度。仍按2ms物理、20ms策略周期执行，不跳过物理步；默认关闭阴影和反射以减少绘制负担。窗口卡顿与策略上台后不停属于两个独立问题。

当前默认是**第20轮seed71终局ONNX**（2026-11-03同步），第14/13轮seed71保留作对照。第20轮在Isaac的64次评估中，60次双前轮合法落顶、48次四轮上台、45次归髋、36次完成规定短行，**完整停稳仍为0**。第21轮仍在训练，没有发布未完成检查点。

**这次同步不代表MuJoCo效果提高。** 本机严格验证约0.56秒发生前轮碰沿；继续预览时机器人可以向前越过台阶，但没有合法抬轮越沿序列，仍会重复抬轮并持续前进。15秒运行超时，未停车，窗口显示`TASK FAILED`。默认展示最新已审查训练策略；若需与上次桌面体验比较，使用`--policy policies/round14_seed71`。旧第14轮用户反馈可登台但不停且多余抬头，同样没有完整任务验收。没有添加规则停车或手写登台动作。

## 安装与运行

在主仓库根目录同步独立项目：

```bash
repo sync rl_project/std_s2s_step_project rl_project/std_train_step_project
cd rl_project/std_s2s_step_project
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python run.py
```

桌面需Python 3.10及以上和OpenGL图形环境。运行端只使用NumPy、MuJoCo、ONNX Runtime CPU，不需要Isaac、PyTorch、CUDA、训练demo或PT检查点。默认相对项目位置寻找主仓库的只读`robot_description/`；也可用`--model-dir`指定完整模型目录。策略和模型哈希不匹配时拒绝启动。

初始等待时暂停物理和策略计算，保留训练的默认初态。Space只从初始等待状态启动，运行中重复按Space不会重新初始化；R可在任何时候重置。默认最多运行15秒，随后停留显示结果；画面暂停不计作机器人停稳。

## 策略与检查方式

```bash
# 上次桌面体验的第14轮对照；第13轮也保留
.venv/bin/python run.py --policy policies/round14_seed71

# 严格模式：第一个任务失败时暂停画面，便于定位
.venv/bin/python run.py --stop-on-failure

# 无窗口真实物理闭环，保存实际请求、执行、关节和任务轨迹
.venv/bin/python run.py --headless --start --seconds 6 --output outputs/preview
```

输出目录必须尚不存在，避免覆盖已有证据。默认预览会锁存第一次失败并继续执行同一策略；失败后不会重新记成成功。失败码：1前轮触边/立面，2后轮离地，3倾角或机身接触保护，4倒退/横移越界，5前轮重复离地或抬起后回落原地面（第17轮及以后）。只有原严格完成判据满足才显示`COMPLETE`，不能用运动画面替代验收。

`policies/selection.json`记录选择依据；每份`policy.json`记录训练commit、独立run_id、终局检查点、网络/ONNX/模型哈希与Isaac评估。第20轮seed71是本轮两份已审查终局中进度较好的策略，不宣称优于所有旧策略或是MuJoCo最佳。默认策略固定，不在启动时自动下载或选择未评估权重。

## 接口与验证

- 物理步长2ms、策略20ms；默认29维输入为一次性台阶信息、自身IMU/编码器估计26维，加3维传感任务记忆。这3项任务记忆只估计归位后短行/停车意图，不能当作真实完成判据。ONNX包含前26维经验归一化、末3维原始0–1尺度、MLP和Beta均值；旧26维策略仍可运行。
- 两维输出映射为左右共同髋角、共同后轮目标速度；第20轮轮速使用`25*clip((a+0.9)/1.9,0,1)`，执行目标反馈仍为`wheel/12.5-1`。沿用训练目标变化率、PD和源执行器限幅，头耳保持默认目标。旧策略按各自元数据选择原映射。
- 台阶高2cm，边沿位于初始整机质心前30cm，垂直行驶方向，平台长/宽均2m。源MJCF/STL及机器人物理参数不修改，只在内存组合环境场景。
- MuJoCo真值只进入独立任务监测和记录，不进入策略观测。默认失败后继续预览不会清除失败标记或添加收尾控制。

最新发布验证见`validation/round20_20261103/review.json`：12,355组ONNX对齐、传感估计/任务记忆移植、真实动作目标重建、2ms单次离地监测重建、模型物理数组不变、真实MuJoCo闭环、静止启动/R复现和实际窗口Space/R事件。运行端不导入PyTorch。程序和接口通过不代表策略行为通过。历史发布证据仍在`validation/review.json`。

维护工具：`tools/export_policy.py`在训练容器CPU上从已完成且已独立评估的运行导出；`tools/validate_port.py`对照对应冻结源码；`tools/validate_memory.py`核对29维策略的传感任务记忆、轮映射和真实子步离地计数；`tools/check_runtime.py`检查真实物理与重置；`tools/check_gui.py`用独立Xvfb和实际X11按键验证GUI（验证端另需`python-xlib`）。日常运行不需要这些验证依赖。MuJoCo接口依据[官方Python文档](https://mujoco.readthedocs.io/en/stable/python.html)和[传感器说明](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-accelerometer)。

本项目通过repo管理，分支`rlproj/std_s2s_step_project`。`std_train_step_project`只保留repo空分支，训练仍由独立step demo维护。

显示修复验证见`validation/display_20261102.json`：1200步传感/复位计算与旧版逐项一致，1500个物理子步对应的150帧完整记录及异步窗口轨迹前缀逐项一致，模型物理、空格/R和真实窗口均通过。共享服务器Xvfb软件渲染从约1.60提高到10.68 FPS，新的仿真循环6.03秒墙钟推进5.90秒物理；这些是同机有限对照，不能保证桌面60 FPS。实际帧率由机器、图形驱动和窗口负载决定。
