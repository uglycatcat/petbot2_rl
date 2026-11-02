# Step：ONNX → MuJoCo 预览

启动后机器人以默认姿态静止显示。按 **Space** 执行策略，按 **R** 重置机器人、策略输入、惯性估计、目标和任务记录。鼠标左键旋转视角，右键平移，滚轮缩放，Esc退出。

2026-11-02显示修复：物理/策略计算与窗口绘制分开，窗口默认目标60 FPS（可用`--render-fps`调整），画面显示实际FPS和仿真速度。仍按2ms物理、20ms策略周期执行，不跳过物理步；默认关闭阴影和反射以减少绘制负担。窗口卡顿与策略上台后不停属于两个独立问题。

当前默认是已完成训练的第14轮seed71终局ONNX；第13轮seed71保留作对照。**目前没有通过完整上台、归髋、短行、停车的sim2sim验收。** 默认候选在MuJoCo中可以抬前轮越沿、落顶，约1.04秒记录到后轮离地；程序会继续展示策略运动，并显示`TASK FAILED`。这是策略预览，不是成功演示，也没有规则停车或手写登台动作。用户桌面反馈确认能完成登台运动，但上台后不停、存在两次抬头（第一次无效）；下一轮训练需加入单次真实离地约束，当前策略包尚未修复这些动作缺陷。

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
# 第13轮对照：Isaac登台/直行较好，但MuJoCo中约0.82秒前轮触边
.venv/bin/python run.py --policy policies/round13_seed71

# 严格模式：第一个任务失败时暂停画面，便于定位
.venv/bin/python run.py --stop-on-failure

# 无窗口真实物理闭环，保存实际请求、执行、关节和任务轨迹
.venv/bin/python run.py --headless --start --seconds 6 --output outputs/preview
```

输出目录必须尚不存在，避免覆盖已有证据。默认预览会锁存第一次失败并继续执行同一策略；失败后不会重新记成成功。失败码：1前轮触边/立面，2后轮离地，3倾角或机身接触保护，4倒退/横移越界。只有原严格完成判据满足才显示`COMPLETE`，不能用运动画面替代验收。

`policies/selection.json`记录选择依据；每份`policy.json`记录训练commit、独立run_id、终局检查点、网络/ONNX/模型哈希与Isaac评估。选择时第15轮仍在运行，未把未评估的中间检查点称为最佳。第14轮在本次MuJoCo对照中前轮动作更完整，第13轮仍是Isaac进度更好的对照；都没有完整成功。

## 接口与验证

- 物理步长2ms、策略20ms；26维输入保持一次性台阶信息及自身IMU/编码器估计。ONNX包含训练观测归一化、MLP和Beta均值输出。
- 两维输出映射为左右共同髋角、共同后轮目标速度；沿用训练目标变化率、PD和源执行器限幅。头耳保持默认目标。
- 台阶高2cm，边沿位于初始整机质心前30cm，垂直行驶方向，平台长/宽均2m。源MJCF/STL及机器人物理参数不修改，只在内存组合环境场景。
- MuJoCo真值只进入独立任务监测和记录，不进入策略观测。默认失败后继续预览不会清除失败标记或添加收尾控制。

发布验证见`validation/review.json`：ONNX数值对齐、传感估计/观测移植对齐、48,000条原记录的目标限幅重建、模型物理数组不变、实际MuJoCo闭环、启动静止、R后重复轨迹、真实窗口Space/R事件。测试支持程序与接口有效，当前策略仍存在跨引擎动作失败。

维护工具：`tools/export_policy.py`在训练容器CPU上从已完成且已独立评估的运行导出；`tools/validate_port.py`对照对应冻结源码；`tools/check_runtime.py`检查真实物理与重置；`tools/check_gui.py`用独立Xvfb和实际X11按键验证GUI（验证端另需`python-xlib`）。日常运行不需要这些验证依赖。MuJoCo接口依据[官方Python文档](https://mujoco.readthedocs.io/en/stable/python.html)和[传感器说明](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-accelerometer)。

本项目通过repo管理，分支`rlproj/std_s2s_step_project`。`std_train_step_project`只保留repo空分支，训练仍由独立step demo维护。

显示修复验证见`validation/display_20261102.json`：1200步传感/复位计算与旧版逐项一致，1500个物理子步对应的150帧完整记录及异步窗口轨迹前缀逐项一致，模型物理、空格/R和真实窗口均通过。共享服务器Xvfb软件渲染从约1.60提高到10.68 FPS，新的仿真循环6.03秒墙钟推进5.90秒物理；这些是同机有限对照，不能保证桌面60 FPS。实际帧率由机器、图形驱动和窗口负载决定。
