"""Headless check that the rear-wheel balance actually stops."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np

from attitude import read_rpy
from control import JumpController, nose_up
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
        self.assertLess(float(np.max(np.abs(vx[tail]))), 0.15)
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
            self.assertEqual(str(gui._jump["state"]), "disabled")
            self.assertEqual(str(gui._cross["state"]), "disabled")
            self.assertEqual(gui._jump["text"], "Rear Hop")
            self.assertEqual(gui._cross["text"], "Cross")
            gui.set_actions_enabled(True)
            gui._request_jump()
            gui._request_cross()
            self.assertFalse(gui.consume_jump())
            self.assertTrue(gui.consume_cross())
            self.assertFalse(gui.consume_cross())
        finally:
            gui.close()


class RearHopButtonTest(unittest.TestCase):
    def test_hop_waits_until_steady_then_restarts_on_four_wheels(self):
        model, data, controller, dt = _world()
        self.assertFalse(controller.request_jump())
        while data.time < 12.0 and not controller.jump_ready:
            controller.step(data, dt)
            mujoco.mj_step(model, data)
        self.assertTrue(controller.jump_ready)
        self.assertEqual(controller.phase, "hold")
        started = data.time
        self.assertTrue(controller.request_jump())
        self.assertFalse(controller.jump_ready)
        controller.step(data, dt)
        limit = controller.hop_torque
        self.assertAlmostEqual(float(data.ctrl[controller.act["left_frame"]]), limit)
        self.assertAlmostEqual(float(data.ctrl[controller.act["right_frame"]]), limit)
        self.assertEqual(controller.phase, "hop")

        while data.time < started + 8.0 and controller.phase == "hop":
            self.assertFalse(controller.jump_ready)
            controller.step(data, dt)
            mujoco.mj_step(model, data)
        self.assertEqual(controller.phase, "drive")
        self.assertGreater(data.time, started)
        self.assertFalse(controller.request_jump())
        self.assertFalse(controller.request_cross())


class CrossButtonTest(unittest.TestCase):
    def test_cross_full_wheel_speed_pitches_the_nose_forward(self):
        model, data, controller, dt = _world()
        while data.time < 12.0 and not controller.jump_ready:
            controller.step(data, dt)
            mujoco.mj_step(model, data)
        self.assertTrue(controller.jump_ready)
        pitch0 = nose_up(data.qpos)
        self.assertTrue(controller.request_cross())
        self.assertFalse(controller.request_jump())
        self.assertEqual(controller.phase, "cross")

        limit = -controller.hop_torque
        reverse = -controller.full_wheel_speed
        forward = controller.full_wheel_speed
        while controller._phase_time < 0.20 and controller.phase == "cross":
            controller.step(data, dt)
            mujoco.mj_step(model, data)
            used = controller._phase_time - dt
            wheel = reverse if used < 0.10 else forward
            self.assertTrue(np.isnan(controller.frame_target).all())
            self.assertAlmostEqual(float(data.ctrl[controller.act["left_frame"]]), limit)
            self.assertAlmostEqual(float(data.ctrl[controller.act["right_frame"]]), limit)
            self.assertAlmostEqual(float(data.ctrl[controller.act["left_rear_wheel"]]), wheel)
            self.assertAlmostEqual(float(data.ctrl[controller.act["right_rear_wheel"]]), wheel)
        self.assertGreater(controller._phase_time, 0.15)
        self.assertLess(nose_up(data.qpos), pitch0 - 0.4)


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
