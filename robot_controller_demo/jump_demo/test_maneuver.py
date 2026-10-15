"""Headless check that the rear-wheel balance actually stops."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np

from attitude import read_rpy
from control import JumpController
from gui import FRAME_SERIES, IMU_SERIES, WHEEL_SERIES, JumpGui, TraceLog


class JumpManeuverTest(unittest.TestCase):
    def test_drive_lift_and_stop(self):
        model_path = Path(__file__).resolve().parents[2] / "robot_description/mjcf/scene.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        data = mujoco.MjData(model)
        controller = JumpController(model)
        mujoco.mj_forward(model, data)
        dt = model.opt.timestep
        horizon = 12.0
        steps = int(horizon / dt)
        x = np.empty(steps)
        vx = np.empty(steps)
        frame = np.empty(steps)
        front = np.empty(steps)
        rear = np.empty(steps)
        for i in range(steps):
            controller.step(data, dt)
            mujoco.mj_step(model, data)
            x[i] = data.qpos[0]
            vx[i] = data.qvel[0]
            frame[i] = data.qpos[controller.qadr["left_frame"]]
            front[i], rear[i] = controller.wheel_height(data)

        drive = x[: int(2.0 / dt)]
        self.assertGreater(float(drive.max()), 0.30)
        self.assertGreater(float(frame.max()), 0.9)
        self.assertLess(float(frame.min()), -0.35)

        airborne = (front > 0.12) & (rear < 0.042)
        run = 0
        longest = 0
        for up in airborne:
            run = run + 1 if up else 0
            longest = max(longest, run)
        self.assertGreater(longest * dt, 4.0)

        tail = slice(int(10.5 / dt), None)
        self.assertLess(float(np.max(np.abs(vx[tail]))), 0.08)
        self.assertLess(float(x[tail].max() - x[tail].min()), 0.06)
        self.assertGreater(float(front[tail].min()), 0.12)
        self.assertLess(float(rear[tail].max()), 0.042)

    def test_setpoints_and_imu(self):
        model, data, controller, dt = _world()
        controller.step(data, dt)
        mujoco.mj_step(model, data)
        self.assertEqual(controller.phase, "drive")
        self.assertAlmostEqual(controller.frame_target[0], 0.0)
        self.assertAlmostEqual(controller.frame_target[1], 0.0)
        self.assertAlmostEqual(controller.wheel_target[0], 11.0)
        self.assertAlmostEqual(controller.wheel_target[1], 11.0)
        roll, pitch, yaw = read_rpy(model, data)
        self.assertLess(max(abs(roll), abs(pitch), abs(yaw)), 0.2)

        while data.time < 3.0 and controller.phase != "rear":
            controller.step(data, dt)
            mujoco.mj_step(model, data)
        controller.step(data, dt)
        self.assertEqual(controller.phase, "rear")
        self.assertTrue(np.isnan(controller.frame_target).all())
        self.assertTrue(np.isfinite(controller.wheel_target).all())

    def test_trace_keeps_ten_seconds_and_three_figures_draw(self):
        names = [name for series in (FRAME_SERIES, WHEEL_SERIES, IMU_SERIES) for name, _, _, _ in series]
        log = TraceLog(10.0, 0.002, names)
        for i in range(int(12.0 / 0.002)):
            log.push(i * 0.002, {name: float(i) for name in names})
        time, value = log.window(12.0 - 0.002)
        self.assertGreaterEqual(time[0], 2.0 - 0.01)
        self.assertLess(time[-1] - time[0], 10.01)
        self.assertEqual(value.shape[0], 11)

        gui = JumpGui(0.002)
        try:
            for i in range(20):
                gui.push(i * 0.002, {name: 0.1 * i for name in names})
            gui.render(19 * 0.002, "drive")
            self.assertEqual(len(gui._plots), 3)
            self.assertEqual(gui.root.title(), "PetBot Joint Data")
        finally:
            gui.close()


def _world():
    model_path = Path(__file__).resolve().parents[2] / "robot_description/mjcf/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    controller = JumpController(model)
    mujoco.mj_forward(model, data)
    return model, data, controller, model.opt.timestep


if __name__ == "__main__":
    unittest.main()
