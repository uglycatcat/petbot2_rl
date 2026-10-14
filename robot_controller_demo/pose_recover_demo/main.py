"""Pose-recovery demo: a MuJoCo window and a status window.

Passive releases every actuator. Track holds the three bars. Recover homes the
joints in 2 seconds. Stand-up homes an upright robot in 1 s. A fall with
negative pitch uses the 4 s sequence. A fall with positive pitch runs its
4 s frame motion and then that same sequence. The floating base is left alone.

    python robot_controller_demo/pose_recover_demo/main.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import mujoco.viewer

from attitude import FallJudge, read_imu
from control import DemoState, RecoveryController
from gui import RecoveryGui


def main():
    model_path = Path(__file__).resolve().parents[2] / "robot_description/mjcf/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    controller = RecoveryController(model)
    judge = FallJudge()
    pending_mode = []
    pending_bar = []
    pending_drag = []
    pending_stand = []
    gui = RecoveryGui(
        on_mode=pending_mode.append,
        on_bar=lambda channel, value: pending_bar.append((channel, value)),
        on_drag=pending_drag.append,
        on_stand=lambda: pending_stand.append(True),
        limits=controller.limits,
    )
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = data.xpos[model.body("lower_body").id]
            viewer.cam.distance = 0.72
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -18
        dt = model.opt.timestep
        last = time.monotonic()
        accumulator = 0.0
        next_render = last
        next_ui = last
        try:
            while viewer.is_running() and gui.alive:
                now = time.monotonic()
                accumulator += min(now - last, 0.05)
                last = now
                with viewer.lock():
                    while pending_drag:
                        controller.dragging = pending_drag.pop(0)
                    while pending_bar:
                        channel, value = pending_bar.pop(0)
                        controller.set_bar(channel, value)
                    while pending_mode:
                        controller.set_mode(pending_mode.pop(0), data, force=True)
                    _, pitch_now, _ = read_imu(model, data)
                    while pending_stand:
                        pending_stand.pop(0)
                        controller.request_stand(data, float(pitch_now[1]), judge.fallen)
                    while accumulator >= dt:
                        controller.step(data, dt)
                        mujoco.mj_step(model, data)
                        accumulator -= dt
                    rotation, rpy, tilt = read_imu(model, data)
                    fallen = judge.update(tilt)
                    viewer.cam.lookat[:] = data.xpos[model.body("lower_body").id]
                frame, ear, wheel = controller.command_targets()
                manual = (
                    controller.manual["frame"] is not None,
                    controller.manual["ear"] is not None,
                    controller.manual["wheel"] is not None,
                )
                if controller.standing:
                    progress = controller._stand_elapsed / controller.stand_duration
                else:
                    progress = controller.progress
                state = DemoState(
                    controller.mode, progress, rotation, rpy, tilt, fallen,
                    frame, ear, wheel, manual, controller.dragging, controller.standing,
                    controller.stand_duration if controller.standing else 0.0,
                )
                if now >= next_ui:
                    gui.render(state)
                    next_ui = now + 1 / 30
                gui.pump()
                if now >= next_render:
                    viewer.sync()
                    next_render = now + 1 / 60
                time.sleep(0.002)
        finally:
            gui.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
