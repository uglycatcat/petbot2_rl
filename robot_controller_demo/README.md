# robot_controller_demo

本机 MuJoCo 里的恢复参照和历史键盘驾驶。这里不训练。模型只读，各程序都加载 [`../robot_description/mjcf/scene.xml`](../robot_description/mjcf/scene.xml)，彼此不引用。

图形程序需要本机显示器，以及已安装 MuJoCo 的 Python 环境。

| 路径 | 内容 |
|---|---|
| [pose_recover_demo/](pose_recover_demo/) | 固定姿态恢复。动作组与试操结论见 [固定姿态恢复](../docs/fixed_pose_recovery.md) |
| [jump_demo/](jump_demo/) | 前进、车架配合，前轮离地后以后轮维持倒立摆并停住。结论见 [跳跃预研](../docs/jump_prestudy.md) |
| [test/rear_wheel_inverted_pendulum/](test/rear_wheel_inverted_pendulum/) | 前轮离地的一次约束动力学测试，不开窗口 |
| [test/fall_pose_statistics/](test/fall_pose_statistics/) | 执行器关闭后的被动沉降。结果见同一篇文档的[被动沉降](../docs/fixed_pose_recovery.md#被动沉降) |
| [driver_demo/](driver_demo/README.md) | 早期终端键盘驾驶。历史记录 |

## pose_recover_demo

MuJoCo 窗口加一个英文状态界面。启动后停在 RECOVER，默认站姿被保持。PASSIVE、TRACK、RECOVER 三者互斥。STAND UP 接管当前模式，结束后回到进入前的模式。只有 TRACK 能拖三条 bar：左右车架、左右耳、两个后轮，两侧使用同一带符号目标。

按下 STAND UP 时：未摔倒用 1 秒把关节收回初始角；摔倒且 pitch < 0 走 4 秒动作；摔倒且 pitch ≥ 0 走 8 秒动作。界面不标出是哪一套。完整时间表、增益和摔倒阈值见 [固定姿态恢复](../docs/fixed_pose_recovery.md)。

右侧坐标系是 `lower_body` 上 IMU 的姿态。摔倒看 IMU +Z 与世界 +Z 的夹角：达到 45° 标为 FALLEN，回到 35° 以下才清除。在 MuJoCo 窗口里双击机体，再按住 Ctrl 拖拽，可改变空间姿态。

```bash
python robot_controller_demo/pose_recover_demo/main.py
python -m unittest discover -s robot_controller_demo/pose_recover_demo -q
```

## jump_demo

MuJoCo 窗口跟着机器人。另有一个英文数据窗口，画 10 秒内的车架位置、后轮速度和 IMU 的 roll / pitch / yaw。用本机 `mujoco_env`：

```bash
conda activate mujoco_env
python robot_controller_demo/jump_demo/main.py
python -m unittest robot_controller_demo.jump_demo.test_maneuver robot_controller_demo.test.rear_wheel_inverted_pendulum.test_lift_feasibility -q
```

## 被动沉降统计

`collect_resting_poses.py` 关闭执行器，把机器人以均匀朝向贴地放下，并加入侧躺、倒立、头朝下、头朝上探针，记录停稳后的姿态。1000 套停稳，倾斜角最大 121.7°。说明与分档见 [被动沉降](../docs/fixed_pose_recovery.md#被动沉降)，原始数据在 `test/fall_pose_statistics/output/`。

```bash
python robot_controller_demo/test/fall_pose_statistics/collect_resting_poses.py
```

## driver_demo

方向键驾驶、空格刹车、`c` 切换控腿。头、耳、车架用外部位置 PID 出力矩。说明见 [driver_demo/README.md](driver_demo/README.md)。

```bash
python robot_controller_demo/driver_demo/main.py
```
