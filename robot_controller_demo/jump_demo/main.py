"""Rear-wheel inverted pendulum.

The robot drives forward, the frame joints pitch the body and then fold, the
front wheels leave the ground, and the rear wheels hold an inverted pitch.
Rear Hop and Cross stay off until that stance is steady, and only one can
run. A hop puts both frames at their positive torque limit. A cross drives
both frames to their negative torque limit. The rear wheels run full speed
backward for a moment, then full speed forward, and the nose falls forward.
If all four wheels touch down, the forward roll starts again.

    conda activate mujoco_env
    python robot_controller_demo/jump_demo/main.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import mujoco.viewer
import numpy as np

from attitude import read_rpy
from control import JumpController
from gui import JumpGui


def main():
    model_path = Path(__file__).resolve().parents[2] / "robot_description/mjcf/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    controller = JumpController(model)
    gui = JumpGui(model.opt.timestep)
    mujoco.mj_forward(model, data)

    try:
        with mujoco.viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = data.xpos[model.body("lower_body").id]
                viewer.cam.distance = 0.85
                viewer.cam.azimuth = 140
                viewer.cam.elevation = -18
            dt = model.opt.timestep
            last = time.monotonic()
            accumulator = 0.0
            next_render = last
            next_ui = last
            while viewer.is_running() and gui.alive:
                now = time.monotonic()
                accumulator += min(now - last, 0.05)
                last = now
                with viewer.lock():
                    while accumulator >= dt:
                        if gui.consume_jump():
                            controller.request_jump()
                        elif gui.consume_cross():
                            controller.request_cross()
                        controller.step(data, dt)
                        mujoco.mj_step(model, data)
                        gui.push(float(data.time), _sample(model, data, controller))
                        accumulator -= dt
                    viewer.cam.lookat[:] = data.xpos[model.body("lower_body").id]
                if now >= next_ui:
                    gui.set_actions_enabled(controller.jump_ready)
                    gui.render(float(data.time), controller.phase)
                    next_ui = now + 1 / 30
                gui.pump()
                if now >= next_render:
                    viewer.sync()
                    next_render = now + 1 / 60
                time.sleep(0.001)
    finally:
        gui.close()


def _sample(model, data, controller):
    roll, pitch, yaw = read_rpy(model, data)
    return {
        "left_frame_target": float(controller.frame_target[0]),
        "left_frame": float(data.qpos[controller.qadr["left_frame"]]),
        "right_frame_target": float(controller.frame_target[1]),
        "right_frame": float(data.qpos[controller.qadr["right_frame"]]),
        "left_wheel_target": float(controller.wheel_target[0]),
        "left_wheel": float(data.qvel[controller.dadr["left_rear_wheel"]]),
        "right_wheel_target": float(controller.wheel_target[1]),
        "right_wheel": float(data.qvel[controller.dadr["right_rear_wheel"]]),
        "roll": float(np.degrees(roll)),
        "pitch": float(np.degrees(pitch)),
        "yaw": float(np.degrees(yaw)),
    }


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
