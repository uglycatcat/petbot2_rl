"""Headless checks for the fall flag, the 2 s blend, and actuator release."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco

from attitude import FALLEN_ENTER_RAD, FALLEN_EXIT_RAD, FallJudge, rotation_to_rpy, tilt_from_world_up
from control import EAR, FRAME, PASSIVE, RECOVER, TRACK, WHEEL, blend, RecoveryController
from gui import _frame_segments, _pair_readout, bar_value


SCENE = Path(__file__).resolve().parents[2] / "robot_description/mjcf/scene.xml"


def load():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return model, mujoco.MjData(model)


class AttitudeTests(unittest.TestCase):
    def test_identity_is_upright_and_yaw_is_not_a_fall(self):
        identity = np.eye(3)
        self.assertTrue(np.allclose(rotation_to_rpy(identity), 0))
        self.assertAlmostEqual(tilt_from_world_up(identity), 0)
        yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rpy = rotation_to_rpy(yaw)
        self.assertAlmostEqual(rpy[0], 0)
        self.assertAlmostEqual(rpy[1], 0)
        self.assertAlmostEqual(rpy[2], np.pi / 2)
        self.assertAlmostEqual(tilt_from_world_up(yaw), 0)
        judge = FallJudge()
        self.assertFalse(judge.update(0.0))

    def test_roll_is_fallen_and_the_flag_has_hysteresis(self):
        roll = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        self.assertAlmostEqual(tilt_from_world_up(roll), np.pi / 2, places=6)
        judge = FallJudge()
        self.assertFalse(judge.update(FALLEN_ENTER_RAD - 0.01))
        self.assertTrue(judge.update(FALLEN_ENTER_RAD))
        self.assertTrue(judge.update(FALLEN_EXIT_RAD + 0.05))
        self.assertFalse(judge.update(FALLEN_EXIT_RAD))
        self.assertFalse(judge.update(FALLEN_EXIT_RAD + 0.05))

    def test_imu_z_projects_above_the_origin(self):
        segments = _frame_segments(np.eye(3), 384, 228)
        origin_y = segments[0]["origin"][1]
        z_tip = segments[2]["end"]
        self.assertLess(z_tip[1], origin_y)
        tips = {tuple(np.round(segment["end"][:2], 3)) for segment in segments[3:]}
        self.assertEqual(len(tips), 3)


class BlendTests(unittest.TestCase):
    def test_endpoints_are_exact_and_the_middle_is_halfway(self):
        start = np.array([0.4, -0.2, 1.0])
        home = np.zeros(3)
        position, velocity = blend(start, home, 0.0, 2.0)
        self.assertTrue(np.allclose(position, start))
        self.assertTrue(np.allclose(velocity, 0))
        position, velocity = blend(start, home, 2.0, 2.0)
        self.assertTrue(np.allclose(position, home))
        self.assertTrue(np.allclose(velocity, 0))
        position, _ = blend(start, home, 1.0, 2.0)
        self.assertTrue(np.allclose(position, 0.5 * start))


class ControllerTests(unittest.TestCase):
    def test_commands_never_rewrite_pose(self):
        model, data = load()
        controller = RecoveryController(model)
        data.qpos[3:7] = [0.5, 0.5, 0.5, 0.5]
        data.qpos[controller.qadr] = [0.3, 0.4, -0.2, 0.5, -0.4]
        mujoco.mj_forward(model, data)
        before = data.qpos.copy()
        controller.set_mode(PASSIVE, data)
        controller.step(data, 0.002)
        self.assertTrue(np.allclose(data.qpos, before))
        self.assertTrue(np.allclose(data.ctrl, 0))
        self.assertTrue(np.allclose(model.actuator_gainprm[controller.wheels, 0], 0))
        controller.set_mode(RECOVER, data)
        controller.step(data, 0.002)
        self.assertTrue(np.allclose(data.qpos, before))
        self.assertTrue(np.allclose(data.ctrl[controller.wheels], 0))
        self.assertTrue(np.allclose(model.actuator_gainprm[controller.wheels, 0], controller._gain0))

    def test_recover_returns_joints_without_requiring_an_upright_base(self):
        model, data = load()
        controller = RecoveryController(model)
        data.qpos[2] = 0.16
        data.qpos[3:7] = [np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0, 0.0]
        data.qpos[controller.qadr] = [0.6, 1.0, -0.5, 0.7, -0.8]
        controller.set_mode(PASSIVE, data)
        controller.set_mode(RECOVER, data)
        dt = model.opt.timestep
        steps = round(2.4 / dt)
        for _ in range(steps):
            controller.step(data, dt)
            mujoco.mj_step(model, data)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr] - controller.home)), 0.05)
        tilt = np.arccos(np.clip(data.site_xmat[model.site("imu").id].reshape(3, 3)[2, 2], -1, 1))
        self.assertGreater(tilt, np.deg2rad(45))

    def test_passive_releases_the_frames(self):
        model, data = load()
        controller = RecoveryController(model)
        controller.set_mode(PASSIVE, data)
        dt = model.opt.timestep
        for _ in range(round(1.2 / dt)):
            controller.step(data, dt)
            mujoco.mj_step(model, data)
        frames = data.qpos[controller.qadr[3:]]
        self.assertLess(frames[0], -0.5)
        self.assertLess(frames[1], -0.5)

    def test_holding_home_keeps_the_default_stance(self):
        model, data = load()
        controller = RecoveryController(model)
        dt = model.opt.timestep
        for _ in range(round(2.0 / dt)):
            controller.step(data, dt)
            mujoco.mj_step(model, data)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr] - controller.home)), 0.02)
        tilt = np.arccos(np.clip(data.site_xmat[model.site("imu").id].reshape(3, 3)[2, 2], -1, 1))
        self.assertLess(tilt, np.deg2rad(2))


class BarTests(unittest.TestCase):
    def test_slider_ends_match_the_joint_range(self):
        low, high = -1.0, 1.2
        self.assertAlmostEqual(bar_value(7, 384, low, high), low)
        self.assertAlmostEqual(bar_value(384 - 7, 384, low, high), high)
        self.assertAlmostEqual(_pair_readout(0.4, 0.4, "rad"), _pair_readout(0.4, 0.402, "rad"))
        self.assertIn("L", _pair_readout(0.4, -0.2, "rad"))

    def _run(self, controller, data, seconds):
        dt = controller._model.opt.timestep
        for _ in range(round(seconds / dt)):
            controller.step(data, dt)
            mujoco.mj_step(controller._model, data)

    def test_bars_move_only_in_track(self):
        model, data = load()
        controller = RecoveryController(model)
        controller.set_bar(FRAME, 0.45)
        self.assertIsNone(controller.manual[FRAME])
        controller.set_mode(TRACK, data)
        controller.set_bar(FRAME, 0.45)
        controller.set_bar(EAR, 0.8)
        controller.set_bar(WHEEL, 4.0)
        self._run(controller, data, 0.8)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[3:]] - 0.45)), 0.05)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[1:3]] - 0.8)), 0.08)
        self.assertTrue(np.allclose(data.ctrl[controller.wheels], 4.0))
        before = data.qpos.copy()
        controller.step(data, model.opt.timestep)
        self.assertTrue(np.allclose(data.qpos, before))
        controller.set_mode(RECOVER, data, force=True)
        self.assertFalse(controller.any_manual)

    def test_drag_blocks_stand_and_passive_ignores_bars(self):
        model, data = load()
        controller = RecoveryController(model)
        controller.set_mode(PASSIVE, data)
        controller.set_bar(FRAME, 0.2)
        self.assertIsNone(controller.manual[FRAME])
        controller.dragging = True
        self.assertFalse(controller.request_stand(data, pitch=-0.4, fallen=True))
        controller.dragging = False
        self._run(controller, data, 0.6)
        self.assertLess(data.qpos[controller.qadr[3]], -0.2)

    def test_stand_hits_each_waypoint_and_restores_the_mode(self):
        model, data = load()
        controller = RecoveryController(model)
        controller.set_mode(TRACK, data)
        controller.set_bar(FRAME, 0.2)
        data.qpos[controller.qadr] = [0.4, 0.6, -0.4, 0.5, -0.3]
        mujoco.mj_forward(model, data)
        self.assertTrue(controller.request_stand(data, pitch=-0.4, fallen=True))
        held = controller.manual[EAR]
        controller.set_bar(EAR, 0.9)
        self.assertEqual(controller.manual[EAR], held)
        self._run(controller, data, 1.0)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr] - controller.home)), 0.08)
        self.assertTrue(np.allclose(data.ctrl[controller.wheels], 0.0))
        self._run(controller, data, 1.0)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[0]])), 0.05)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[1:3]] - 1.0)), 0.08)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[3:5]] - 1.2)), 0.08)
        self._run(controller, data, 1.0)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[0]])), 0.05)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[1:3]] - 0.3)), 0.08)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[3:5]] - 1.2)), 0.08)
        self._run(controller, data, 0.25)
        self.assertTrue(np.allclose(data.ctrl[controller.wheels], 5.0))
        self._run(controller, data, 0.5)
        self.assertTrue(np.allclose(data.ctrl[controller.wheels], -7.0))
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[0]])), 0.05)
        self._run(controller, data, 0.3)
        self.assertFalse(controller.standing)
        self.assertEqual(controller.mode, TRACK)
        self.assertAlmostEqual(controller.manual[FRAME], 0.2)
        self.assertAlmostEqual(controller.stand_duration, controller.STAND_NEGATIVE)

    def test_upright_homes_and_a_fall_follows_the_pitch_sign(self):
        model, data = load()
        controller = RecoveryController(model)
        data.qpos[controller.qadr] = [0.3, 0.5, -0.4, 0.4, -0.2]
        mujoco.mj_forward(model, data)
        self.assertTrue(controller.request_stand(data, pitch=0.4, fallen=False))
        self.assertEqual(controller._stand_kind, "home")
        self.assertAlmostEqual(controller.stand_duration, 1.0)
        self._run(controller, data, 1.05)
        self.assertFalse(controller.standing)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr] - controller.home)), 0.08)
        self.assertTrue(np.allclose(data.ctrl[controller.wheels], 0.0))

        controller.request_stand(data, pitch=0.0, fallen=True)
        self.assertEqual(controller._stand_kind, "positive")
        controller.standing = False
        controller.request_stand(data, pitch=-0.2, fallen=True)
        self.assertEqual(controller._stand_kind, "negative")
        controller.standing = False

        data.qpos[controller.qadr] = [0.3, 0.4, 0.4, 0.2, 0.2]
        mujoco.mj_forward(model, data)
        self.assertTrue(controller.request_stand(data, pitch=0.3, fallen=True))
        self.assertAlmostEqual(controller.stand_duration, 8.0)
        self._run(controller, data, 1.0)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr] - controller.home)), 0.08)
        self._run(controller, data, 1.0)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[0]])), 0.05)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[1:3]])), 0.08)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[3:5]] + 1.0)), 0.08)
        self.assertTrue(np.allclose(data.ctrl[controller.wheels], 0.0))
        self._run(controller, data, 1.0)
        self.assertLess(abs(data.qpos[controller.qadr[3]] - 1.0), 0.12)
        self.assertLess(abs(data.qpos[controller.qadr[4]] + 1.0), 0.12)
        self._run(controller, data, 1.0)
        self.assertTrue(controller.standing)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[3:5]] + 0.4)), 0.12)
        self._run(controller, data, 1.0)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr] - controller.home)), 0.08)
        self._run(controller, data, 1.0)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[1:3]] - 1.0)), 0.08)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[3:5]] - 1.2)), 0.08)
        self._run(controller, data, 1.0)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[1:3]] - 0.3)), 0.08)
        self.assertLess(np.max(np.abs(data.qpos[controller.qadr[3:5]] - 1.2)), 0.08)
        self._run(controller, data, 0.25)
        self.assertTrue(np.allclose(data.ctrl[controller.wheels], 5.0))
        self._run(controller, data, 0.5)
        self.assertTrue(np.allclose(data.ctrl[controller.wheels], -7.0))
        self._run(controller, data, 0.3)
        self.assertFalse(controller.standing)
        self.assertEqual(controller.mode, RECOVER)


if __name__ == "__main__":
    unittest.main()
