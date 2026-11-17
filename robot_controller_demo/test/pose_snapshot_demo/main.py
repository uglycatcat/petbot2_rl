"""Manual PetBot pose capture. Run in the desktop's mujoco_env environment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import signal
import sys
import tempfile
import threading
import time

# Select pyGLFW's bundled X11 library before MuJoCo can import it. The native
# Wayland/EGL library on this desktop crashes during viewer teardown.
if sys.platform == "linux" and os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import numpy as np

if __package__:
    from .pose_io import FORMAT, LEGACY_FORMAT, FLOOR_CLEARANCE_M, SNAPSHOTS, PoseLoader
else:
    from pose_io import FORMAT, LEGACY_FORMAT, FLOOR_CLEARANCE_M, SNAPSHOTS, PoseLoader

POSITION_JOINTS = ("head", "left_ear", "right_ear", "left_frame", "right_frame")
# External position control, using the existing model's torque limits.
KP = np.array([8.0, 4.0, 4.0, 12.0, 12.0])
KD = np.array([0.35, 0.04, 0.04, 0.35, 0.35])


class PoseDemo(PoseLoader):
    def __init__(self):
        super().__init__()
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in POSITION_JOINTS])
        self.dadr = np.array([self.model.joint(n).dofadr[0] for n in POSITION_JOINTS])
        self.acts = np.array([self.model.actuator(n).id for n in POSITION_JOINTS])
        self.home = self.model.qpos0[self.qadr].copy()
        self.limits = self.model.actuator_ctrlrange[self.acts].copy()
        self.imu_id = self.model.site("imu").id
        self.refresh()

    def control(self):
        self.data.ctrl[:] = 0.0  # Rear wheel velocity targets remain zero.
        torque = KP * (self.home - self.data.qpos[self.qadr]) - KD * self.data.qvel[self.dadr]
        self.data.ctrl[self.acts] = np.clip(torque, self.limits[:, 0], self.limits[:, 1])

    def refresh(self):
        self.control()
        # mj_step leaves derived quantities at an earlier integration stage.
        # Recompute so IMU readings and the saved qpos describe the same instant.
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        self.control()
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.refresh()

    def imu(self) -> dict:
        rotation = self.data.site_xmat[self.imu_id].reshape(3, 3)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2]))
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        return {
            "quat_wxyz": quat.tolist(),
            "rpy_deg": np.rad2deg([roll, pitch, yaw]).tolist(),
            "tilt_deg": float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
            "gravity_direction_imu": (-rotation[2, :]).tolist(),
            "accel_m_s2": self.data.sensor("imu_accel").data.tolist(),
            "gyro_rad_s": self.data.sensor("imu_gyro").data.tolist(),
        }

    def capture(self) -> dict:
        self.refresh()
        root = self.root_qadr
        qpos, placement = self.prepare_fallen_pose(self.data.qpos)
        return {
            "format": FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mujoco_version": mujoco.__version__,
            "model": self.signature,
            "sim_time_s": float(self.data.time),
            "qpos": qpos.tolist(),
            "qvel_at_capture": self.data.qvel.tolist(),
            "root_height_m": placement["height_m"],
            "captured_root_height_m": placement["source_height_m"],
            "root_quat_wxyz": qpos[root + 3:root + 7].tolist(),
            "joint_position_rad": {
                self.model.joint(i).name: float(self.data.qpos[self.model.jnt_qposadr[i]])
                for i in range(self.model.njnt) if i != self.model.joint("root").id
            },
            "position_targets_rad": dict(zip(POSITION_JOINTS, self.home.tolist())),
            "imu": self.imu(),
            "restore_mode": "fallen_pose_centered_floor_safe_zero_velocity",
            "placement": {
                "horizontal_position": "model_default",
                "minimum_floor_clearance_m": FLOOR_CLEARANCE_M,
                "height_adjustment_m": placement["height_adjustment_m"],
            },
        }

    def save(self, directory: Path) -> Path:
        record = self.capture()
        encoded = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        # Atomic publication; a partial write must not become the latest snapshot.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".writing_", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        path = directory / f"pose_{stamp}_{temporary.stem.removeprefix('.writing_')}.json"
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load(self, path: Path):
        placement = super().load(path)
        self.refresh()
        return placement


def latest_snapshot(directory: Path) -> Path:
    paths = sorted(directory.glob("pose_*.json"))
    if not paths:
        raise FileNotFoundError(f"尚无快照：{directory}")
    return paths[-1]


def clear_drag(viewer, demo):
    viewer.perturb.active = 0
    viewer.perturb.active2 = 0
    demo.data.xfrc_applied[:] = 0.0
    demo.data.qfrc_applied[:] = 0.0


def loaded_message(placement):
    return (f"已加载并暂停：水平归中，高度 {placement['height_m'] * 100:.2f} cm；"
            f"防穿地上移 {placement['height_adjustment_m'] * 1000:.2f} mm。")


@contextmanager
def queued_interrupt(commands):
    # Avoid raising KeyboardInterrupt in the middle of a native viewer call.
    previous = signal.signal(signal.SIGINT, lambda *_: commands.put(("quit", None)))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@contextmanager
def desktop_viewer(model, data):
    import mujoco.viewer

    threads_before = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    # This standalone demo starts no other threads. launch_passive's rendering
    # thread is a daemon, and Handle.close() only requests asynchronous shutdown.
    viewer_threads = set(threading.enumerate()) - threads_before
    try:
        yield viewer
    finally:
        viewer.close()
        for thread in viewer_threads:
            thread.join()  # Finish rendering before Python/GLFW teardown.


def run(args):
    if __package__:
        from .gui import SnapshotPanel
    else:
        from gui import SnapshotPanel

    demo = PoseDemo()
    paused = args.paused
    loaded_path = None
    if args.load:
        path = latest_snapshot(args.snapshot_dir) if args.load == "latest" else Path(args.load)
        placement = demo.load(path)
        paused = True
        loaded_path = path
        print(f"{loaded_message(placement)}\n文件：{path}", flush=True)
    commands = SimpleQueue()

    print("请使用“PetBot 姿态快照”面板保存、加载、复位及暂停仿真。\n"
          "在 MuJoCo 窗口双击机体，Ctrl + 鼠标左键拖动旋转，Ctrl + 右键拖动施力。\n"
          f"快照目录：{args.snapshot_dir}", flush=True)
    with queued_interrupt(commands), SnapshotPanel(commands, args.snapshot_dir) as panel, \
            desktop_viewer(demo.model, demo.data) as viewer:
        if loaded_path:
            panel.show_message(loaded_message(placement), path=loaded_path)
        with viewer.lock():
            viewer.cam.lookat[:] = demo.data.xpos[demo.model.body("lower_body").id]
            viewer.cam.distance = 0.85
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
        previous = time.monotonic()
        accumulator = 0.0
        dt = float(demo.model.opt.timestep)
        stopping = False
        while viewer.is_running() and panel.alive:
            panel.pump()
            if not panel.alive:
                break
            # sync brings the mouse's perturbation forces into the shared data.
            viewer.sync()
            now = time.monotonic()
            if panel.reset_clock:
                previous = now
                accumulator = 0.0
                panel.reset_clock = False
            elapsed = min(now - previous, 0.05)
            previous = now
            with viewer.lock():
                changed_state = False
                while True:
                    try:
                        action, selected_path = commands.get_nowait()
                    except Empty:
                        break
                    if action == "quit":
                        stopping = True
                        break
                    try:
                        if action == "save":
                            path = demo.save(args.snapshot_dir)
                            panel.show_message("摔倒姿态已保存：水平归中，加载时自动防穿地。", path=path)
                            print(f"已保存：{path}", flush=True)
                        elif action == "load":
                            path = selected_path if selected_path is not None else latest_snapshot(args.snapshot_dir)
                            placement = demo.load(path)
                            clear_drag(viewer, demo)
                            paused = True
                            panel.show_message(loaded_message(placement), path=path)
                            changed_state = True
                            print(f"已加载并暂停：{path}", flush=True)
                        elif action == "reset":
                            clear_drag(viewer, demo)
                            demo.reset()
                            panel.show_message("已回到默认位置。")
                            changed_state = True
                        elif action == "pause":
                            paused = not paused
                            clear_drag(viewer, demo)
                            panel.show_message("已暂停，可以保存当前姿态。" if paused else "已继续仿真，可以手动拨动机器人。")
                            changed_state = True
                    except (OSError, ValueError, TypeError) as error:
                        panel.show_message(f"操作失败：{error}", error=True)
                        print(f"快照操作失败：{error}", file=sys.stderr, flush=True)
                if stopping:
                    break
                if paused or changed_state:
                    accumulator = 0.0
                else:
                    accumulator += elapsed
                    while accumulator >= dt:
                        demo.step()
                        accumulator -= dt
                demo.refresh()
                viewer.cam.lookat[:] = demo.data.xpos[demo.model.body("lower_body").id]
            home_error = np.rad2deg(np.max(np.abs(demo.data.qpos[demo.qadr] - demo.home)))
            panel.render(demo.imu(), paused, demo.data.time, home_error,
                         float(demo.data.qpos[demo.root_qadr + 2]))
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.monotonic() - now)))


def main():
    parser = argparse.ArgumentParser(description="PetBot 手动翻倒、IMU/姿态快照与初态恢复")
    parser.add_argument("--load", metavar="JSON_OR_latest", help="加载指定 JSON 或 latest，加载后暂停")
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOTS, help="快照保存目录")
    parser.add_argument("--paused", action="store_true", help="从默认位置暂停启动")
    args = parser.parse_args()
    if sys.platform == "darwin":
        import mujoco.viewer
        if mujoco.viewer._MJPYTHON is None:
            launcher = Path(sys.executable).with_name("mjpython")
            os.execv(str(launcher), [str(launcher), str(Path(__file__).resolve()), *sys.argv[1:]])
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.exit(1, f"启动失败：{error}\n")


if __name__ == "__main__":
    main()
