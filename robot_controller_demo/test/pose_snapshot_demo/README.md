# 手动姿态快照 demo

独立桌面 MuJoCo 测试空间。默认加载原始 `robot_description/mjcf/scene.xml` 的位置和姿态；头、双耳、左右车架以位置 PD 保持模型默认角度，后轮速度目标为零，前轮保持模型原有被动状态。不加载策略，不训练，不修改模型或物理参数，也不引用其他 demo 的代码。

记录目标是**摔倒姿态：旋转、关节构型和高度**。摆姿势产生的水平位移不属于姿态，加载时统一归到默认水平位置；保留高度，仅在地面间隙不足时向上修正。

## 启动与操作

从仓库根目录运行（已在本项目目录时可直接 `python main.py`）：

```bash
conda activate mujoco_env
python robot_controller_demo/test/pose_snapshot_demo/main.py
```

需要 `mujoco`（本机使用 3.9.0）、`numpy`、Python 自带的 `tkinter` 和本机图形环境。启动后出现机器人窗口和一个中文浮动控制面板，快照操作均用按钮完成。
Linux 检测到 DISPLAY 时默认使用 X11/XWayland 窗口后端；选择仅在当前进程生效。退出时等待渲染线程结束，避免 GLFW 与 Python 同时销毁窗口资源。

1. 在 MuJoCo 窗口双击机身，选中后按住 **Ctrl + 鼠标左键拖动**旋转，或 **Ctrl + 右键拖动**施力平移。松开鼠标后可等机器人停稳，再保存。
2. 点击面板的 **保存快照**。每次新建一个 JSON，不覆盖以前的记录；面板显示保存结果和文件名。
3. 点击 **加载最新快照**，或点 **选择快照…** 从文件选择框打开之前的 JSON。加载会水平归中、检查防穿地并自动暂停；面板显示高度与上移量。点击 **继续仿真** 恢复运行。
4. **暂停仿真** 可在保存前冻结当前姿态；**回到默认位置** 恢复模型默认位置和关节状态，保留当前暂停/运行选择。
5. 面板实时显示 IMU 横滚/俯仰/偏航、倾斜角、机身基座高度、加速度、角速度及默认关节角跟踪误差。关闭面板或机器人窗口即可退出。

暂停时不推进动力学，点击 **继续仿真** 后再拖拽。控制面板保持在机器人窗口上方，可拖动到屏幕侧边。本 demo 不绑定操作快捷键；鼠标拨动沿用 [MuJoCo 官方 viewer 操作](https://github.com/google-deepmind/mujoco/blob/3.9.0/simulate/simulate.cc)。

下次启动后可以直接点 **加载最新快照**。也保留命令行直接加载入口：

```bash
python robot_controller_demo/test/pose_snapshot_demo/main.py --load latest
```

也可以选之前的文件，路径按当前终端目录解析：

```bash
python robot_controller_demo/test/pose_snapshot_demo/main.py --load robot_controller_demo/test/pose_snapshot_demo/snapshots/pose_实际文件名.json
```

`--snapshot-dir 路径` 可指定保存目录；默认始终为本 demo 内的 `snapshots/`，不随启动目录改变。`--paused` 从默认状态暂停启动。

## 其他程序直接加载姿态

公共接口为 [`pose_io.py`](pose_io.py) 的 `PoseLoader` 和 `recorded_pose_path()`，只依赖 MuJoCo / NumPy，**不启动窗口、线程、控制器或仿真循环**。四份原始 JSON 及 [`snapshots/manifest.json`](snapshots/manifest.json) 随本 test 项目一起版本管理；固定编号按录制时间排列，不使用会随新录制变化的 `latest`。

| 固定编号 | 原始 JSON |
|---|---|
| `pose_01` | `pose_20261117T113831_986420Z_l7x5slh2.json` |
| `pose_02` | `pose_20261117T113953_350782Z_441crrkq.json` |
| `pose_03` | `pose_20261117T114056_589513Z__q1lqz9v.json` |
| `pose_04` | `pose_20261117T114246_876400Z_8mnowzob.json` |

### 新程序：创建模型并加载

从仓库根目录运行 Python，使仓库根目录位于模块搜索路径中：

```python
from robot_controller_demo.test.pose_snapshot_demo.pose_io import (
    PoseLoader, recorded_pose_path,
)

loader = PoseLoader()
placement = loader.load(recorded_pose_path("pose_03"))
model, data = loader.model, loader.data

# 此刻 data 已是指定摔倒初态；尚未执行任何 mj_step。
print(data.qpos)       # x=y=0，旋转与关节角保持原记录，高度已防穿地
print(placement)       # 原高度、加载高度、上移量、最小地面间隙，单位米
# 在这里交给你自己的 viewer、控制器或仿真循环。
```

`recorded_pose_path()` 返回相对模块定位的绝对文件路径，并核对 SHA-256；不依赖当前工作目录。也可直接传入自己保存的文件：`loader.load("自己的快照.json")`，支持 v1/v2。自定义文件的相对路径按当前工作目录解析，模型校验仍执行。

### 已有程序：直接恢复到自己的 model/data

```python
import mujoco
from robot_controller_demo.test.pose_snapshot_demo.pose_io import (
    SCENE, PoseLoader, recorded_pose_path,
)

model = mujoco.MjModel.from_xml_path(str(SCENE))  # 或使用程序已有的同一原始模型
data = mujoco.MjData(model)
loader = PoseLoader(model=model, data=data)
placement = loader.load(recorded_pose_path("pose_01"))
# loader.data 就是传入的 data，不会换成另一份。
```

调用契约：

- 适用于原始 `robot_description/mjcf/scene.xml`，地面 z=0。传入的 `data` 必须由传入的 `model` 创建；不要用于改过关节、物理参数、地面或新增物体的场景。检查内容是源码 MJCF 摘要、状态维数和关节顺序，不是任意自定义模型的完整等价性证明。
- 调用前停止其他线程修改 `data`；有 passive viewer 时在 `viewer.lock()` 内调用，并清除调用方自己的鼠标扰动力。接口不会暂停外部循环或操作外部 viewer。
- `load()` 先在独立数据上校验、归中、检查地面，再对目标执行 `mj_resetData`、写入整理后的 `qpos` 和 `mj_forward`。时间、速度、控制量、外力及旧求解器状态都会重置；不重放 JSON 的 `qvel_at_capture` 或原始 IMU 读数。验证失败时目标数据不变。
- 调用成功后，原四元数（wxyz，包括偏航）和所有关节角保持不变；x/y 为模型默认值（0/0）；z 只在必要时抬高，初始几何最小地面间隙至少 1 mm。不执行积分步，因此姿态保持到调用方下一次 `mj_step` 为止。
- 接口不带本 demo 的关节 PD 控制。若调用方要继续仿真并保持默认关节目标，由调用方设置控制；本 demo 的 `PoseDemo` 在共享加载接口之后另行更新自己的 PD。MuJoCo 中短暂的接触压入与后续运动由动力学决定，1 mm 保证针对加载初态。
- 别的物理引擎不能直接套 MuJoCo 的 `qpos` 索引；应按 JSON 的关节名称映射角度、确认四元数 wxyz 与坐标轴约定，并在目标引擎重新检查地面间隙。

### 可直接运行的无窗口例子

[`load_pose_example.py`](load_pose_example.py) 是上面接口的完整调用例子，只读取文件并打印加载结果：

```bash
conda activate mujoco_env
python robot_controller_demo/test/pose_snapshot_demo/load_pose_example.py --pose pose_01
python robot_controller_demo/test/pose_snapshot_demo/load_pose_example.py --pose pose_04
```

若已在本项目目录中，使用 `python load_pose_example.py --pose pose_02`。其他位置可用相对路径指向这个脚本；若要用包形式导入，把**实际仓库根目录**加入该程序的 Python 搜索路径，不要硬编码本机 `/home/...` 路径。源码和模型资源路径均由模块位置定位。

## 快照含义

新快照为 `petbot-pose-snapshot-v2`，保存前在独立的仿真数据中整理姿态，不改变当前窗口里的机器人。JSON 内容为：

- `imu`：IMU 到世界系的四元数（wxyz）、ZYX 欧拉角（显示顺序 roll/pitch/yaw，度）、倾斜角、IMU 系重力单位方向、加速度计（m/s²）和陀螺仪（rad/s）。坐标轴为 X 前、Y 左、Z 上；欧拉角在俯仰 ±90° 附近有奇异性，恢复用四元数。
- `qpos`：水平坐标归到模型默认值（当前为 x=y=0），保留四元数和全部实际关节角，高度经过防穿地检查。不保存摆姿势产生的水平偏移。
- `root_height_m`：整理后的机身基座高度；`captured_root_height_m` 是采集时的原高度，`placement.height_adjustment_m` 是防穿地上移量。地面为 z=0，单位米。
- `qvel_at_capture`、仿真时间、UTC 保存时间和 MJCF 校验信息，用于记录当时状态。

加载新旧快照时统一执行：**水平归中 → 保留旋转及关节角 → 保留高度或仅向上修正 → 清零速度、外力和时间 → 暂停**。朝向包含原来的偏航，不额外旋转机身。防穿地用机器人各碰撞几何体到地面的实际有符号距离检查，保证初始最小间隙至少 1 mm；不降低本来就足够的高度，不改变关节角，不修改模型。重复加载/保存不会不断抬高。

旧版 `petbot-pose-snapshot-v1` 直接兼容，其中 x/y 仅作为原始记录留在文件里，加载时忽略。程序不迁移、不覆盖旧文件，无需重新录制。加载计算失败时保留当前窗口里的姿态。

保存的 IMU 加速度/角速度属于采集时刻，加载后的实时传感器数据会按当前动力学重算，不重放拨动冲量。继续仿真后机器人会受重力和接触影响；1 mm 的初始间隙会自然落下，不稳定姿态可能继续变化。

位置保持使用模型现有电机及力矩限制，**并非把关节焊死**；强推或接触时可能有跟踪误差，界面显示五个位置关节的最大默认角误差。快照保留实际关节角，不擅自改回零导致机体穿地。前轮为被动轮，轮子转角也会随接触变化。

以下四份原始 JSON 与 `snapshots/manifest.json` 已按用户要求纳入本次提交范围；`.gitignore` 逐文件放行，后续新录制默认仍忽略。`protected_poses/` 的额外只读副本保持本地，不重复提交。**四份原件及备份为重要数据，不得覆盖、删除或用作测试输出。**

## 四份重要原始姿态（2026-11-17）

原件在 `snapshots/`，以原始字节纳入 Git；[版本化清单](snapshots/manifest.json) 保存固定ID、文件名、SHA-256、大小和录制时间。额外本地只读副本在 `protected_poses/20261117_first_four/`，含原始备份清单；新检出的工作区不要求有此目录，加载程序只依赖已版本化的原件和清单。维护或清理时必须保全重要原件及已有备份。

以下为当前模型的加载检查结果，未写回原始 JSON：

| 原始文件 | 倾斜角 | 加载基座高度 | 防穿地上移 |
|---|---:|---:|---:|
| `pose_20261117T113831_986420Z_l7x5slh2.json` | 102.53° | 9.458 cm | 1.006 mm |
| `pose_20261117T113953_350782Z_441crrkq.json` | 117.09° | 12.570 cm | 1.054 mm |
| `pose_20261117T114056_589513Z__q1lqz9v.json` | 110.94° | 7.595 cm | 1.499 mm |
| `pose_20261117T114246_876400Z_8mnowzob.json` | 109.09° | 10.284 cm | 1.003 mm |

四份均保留原四元数和全部关节角，加载后 x=y=0、最小地面间隙 1 mm。原件及备份 SHA-256 在加载和UI验证后再次核对一致。

## 开发验证

```bash
python -m unittest discover -s robot_controller_demo/test/pose_snapshot_demo -p 'test_*.py' -v
```
