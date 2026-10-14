"""Run: python -m unittest discover -s robot_controller_demo/driver_demo -v"""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from control import DriveCommand, DriveConfig, RobotController, configure_simulation
from keyboard_macos import MacTerminalKeyboard


SCENE = Path(__file__).resolve().parents[2] / 'robot_description/mjcf/scene.xml'


class DrivingTests(unittest.TestCase):
    def test_ramp_limits_and_reverse_brakes_through_zero(self):
        drive = DriveCommand(DriveConfig())
        previous = 0
        for _ in range(1000):
            drive.update({'up', 'left'}, .002)
            self.assertLessEqual(drive.speed - previous, .35 * .002 + 1e-12)
            self.assertLessEqual(drive.speed, .35)
            self.assertLessEqual(drive.yaw_rate, 1.5)
            previous = drive.speed
        drive.update({'down'}, .002)
        self.assertGreater(drive.speed, 0)
        for _ in range(1000):
            drive.update({'down'}, .002)
        self.assertAlmostEqual(drive.speed, -.35)

    def test_release_focus_loss_and_opposite_keys_stop(self):
        for keys, active in [(set(), True), ({'up'}, False),
                             ({'up', 'down', 'left', 'right'}, True),
                             ({'up', 'space'}, True)]:
            drive = DriveCommand(DriveConfig())
            for _ in range(500):
                drive.update({'up', 'left'}, .002)
            for _ in range(500):
                drive.update(keys, .002, active)
            self.assertEqual(drive.speed, 0)
            self.assertEqual(drive.yaw_rate, 0)

    def test_left_and_right_wheel_signs(self):
        drive = DriveCommand(DriveConfig())
        left, right = drive.update({'left'}, .01)
        self.assertLess(left, 0)
        self.assertGreater(right, 0)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        configure_simulation(self.model)
        self.data = mujoco.MjData(self.model)
        self.controller = RobotController(self.model)

    def run_for(self, seconds, keys=frozenset(), active=True):
        for _ in range(round(seconds / self.model.opt.timestep)):
            self.controller.step(self.data, keys, self.model.opt.timestep, active)
            mujoco.mj_step(self.model, self.data)
        self.assertTrue(np.isfinite(self.data.qpos).all())
        self.assertTrue(np.isfinite(self.data.qvel).all())

    def test_pid_recovers_from_joint_displacement(self):
        self.data.qpos[self.controller.qpos] = [.15, .2, -.2, .08, -.08]
        self.run_for(10)
        self.assertLess(np.max(np.abs(self.data.qpos[self.controller.qpos])), .01)
        self.assertGreater(self.data.qpos[2], .07)

    def test_forward_reverse_and_focus_braking_in_scene(self):
        self.run_for(3)
        self.run_for(3, {'up'})
        self.assertGreater(self.data.qpos[0], .7)
        self.run_for(2, {'up'}, active=False)
        self.assertLess(np.linalg.norm(self.data.qvel[:2]), .01)
        start_x = self.data.qpos[0]
        self.run_for(3, {'down'})
        self.assertLess(self.data.qpos[0] - start_x, -.7)
        self.assertLess(np.max(np.abs(self.data.qpos[self.controller.qpos])), .015)

    def test_torque_limits_and_integrator_antiwindup(self):
        self.data.qpos[self.controller.qpos] = 10
        for _ in range(10000):
            self.controller.step(self.data, set(), .002)
        self.assertTrue(np.all(self.data.ctrl[self.controller.actuators] >= self.controller.low))
        self.assertTrue(np.all(self.data.ctrl[self.controller.actuators] <= self.controller.high))
        np.testing.assert_array_equal(self.controller.integral, 0)

    def test_configuration_preserves_mjcf_contact_parameters(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        friction = model.geom_friction.copy()
        priority = model.geom_priority.copy()
        configure_simulation(model)
        np.testing.assert_array_equal(model.geom_friction, friction)
        np.testing.assert_array_equal(model.geom_priority, priority)

    def test_mjcf_alone_has_front_friction_point_one(self):
        # No configure_simulation: contacts must get 0.1 from the XML itself.
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        seen = set()
        for contact in data.contact:
            names = {model.geom(int(g)).name for g in contact.geom}
            front = names & {'left_front_wheel', 'right_front_wheel'}
            if 'floor' in names and front:
                np.testing.assert_allclose(contact.friction[:2], .1)
                seen.update(front)
        self.assertEqual(seen, {'left_front_wheel', 'right_front_wheel'})

    def test_actual_front_contacts_have_low_friction_rears_keep_grip(self):
        self.run_for(2)
        seen = set()
        for contact in self.data.contact:
            names = {self.model.geom(int(g)).name for g in contact.geom}
            if 'floor' not in names:
                continue
            wheel = next((n for n in names if 'wheel' in n), None)
            if wheel is None:
                continue
            seen.add(wheel)
            expected = .1 if 'front' in wheel else 1.5
            np.testing.assert_allclose(contact.friction[:2], expected)
        self.assertEqual(seen, {'left_front_wheel', 'right_front_wheel',
                                'left_rear_wheel', 'right_rear_wheel'})


class FocusTests(unittest.TestCase):
    def test_physical_release_and_other_terminal_tab(self):
        keyboard = MacTerminalKeyboard.__new__(MacTerminalKeyboard)
        keyboard.fd = 0
        keyboard.target = 'original-terminal-pane'
        keyboard.previous = set()
        keyboard.CF = Mock()
        keyboard.CF.CFEqual.side_effect = lambda a, b: a == b
        keyboard.CG = Mock()
        down = {126}
        keyboard.CG.CGEventSourceKeyState.side_effect = lambda state, key: key in down
        keyboard._focused_element = Mock(return_value=keyboard.target)
        with patch('keyboard_macos.select.select', return_value=([], [], [])):
            self.assertEqual(keyboard.poll()[0], {'up'})
            down.clear()  # No terminal repeat/timeout is involved in release detection.
            self.assertEqual(keyboard.poll()[0], set())
            down.add(126)
            keyboard._focused_element.return_value = 'another-terminal-pane'
            keys, _, focused = keyboard.poll()
            self.assertEqual(keys, set())
            self.assertFalse(focused)
            keyboard._focused_element.return_value = keyboard.target
            self.assertEqual(keyboard.poll()[0], {'up'})


if __name__ == '__main__':
    unittest.main()
