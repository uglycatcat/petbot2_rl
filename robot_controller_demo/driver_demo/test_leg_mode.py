"""Mode transitions, frame angle commands and simulated position tracking."""
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from control import RobotController, configure_simulation
from keyboard_macos import MacTerminalKeyboard


class LegModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene = Path(__file__).resolve().parents[2] / 'robot_description/mjcf/scene.xml'
        cls.model = mujoco.MjModel.from_xml_path(str(scene))
        configure_simulation(cls.model)

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        self.controller = RobotController(self.model)

    def step(self, keys=(), dt=.01, active=True):
        self.controller.step(self.data, set(keys), dt, active=active)

    def test_normal_mode_ignores_leg_keys_and_toggle_respects_focus(self):
        self.step({'q', 'r', 'up'})
        np.testing.assert_array_equal(self.controller.frame_targets, 0)
        self.assertGreater(self.controller.drive.speed, 0)
        self.controller.handle_pressed({'c'}, active=False)
        self.assertFalse(self.controller.leg_mode)
        self.controller.handle_pressed({'c'})
        self.assertTrue(self.controller.leg_mode)
        for _ in range(100):
            self.controller.handle_pressed(set())
            self.step({'c'})
        self.assertTrue(self.controller.leg_mode)

    def test_each_key_controls_only_its_joint_in_the_expected_direction(self):
        for key, expected in [('q', [-.005, 0]), ('w', [.005, 0]),
                              ('e', [0, -.005]), ('r', [0, .005])]:
            with self.subTest(key=key):
                self.controller = RobotController(self.model)
                self.controller.handle_pressed({'c'})
                self.step({key})
                np.testing.assert_allclose(self.controller.frame_targets, expected)
                np.testing.assert_array_equal(self.controller.targets[:3], 0)

    def test_combined_driving_and_legs_release_opposites_focus_and_space(self):
        self.controller.handle_pressed({'c'})
        self.step({'q', 'r', 'up', 'left'})
        np.testing.assert_allclose(self.controller.frame_targets, [-.005, .005])
        self.assertGreater(self.controller.drive.speed, 0)
        self.assertGreater(self.controller.drive.yaw_rate, 0)
        target = self.controller.frame_targets.copy()
        for keys, active in [(set(), True), ({'q', 'w', 'e', 'r'}, True),
                             ({'w', 'r'}, False), ({'space', 'w', 'r'}, True)]:
            self.step(keys, active=active)
            np.testing.assert_array_equal(self.controller.frame_targets, target)

    def test_targets_stop_at_mjcf_limits_and_reverse_immediately(self):
        self.controller.handle_pressed({'c'})
        for _ in range(400):
            self.step({'q', 'r'})
        np.testing.assert_allclose(self.controller.frame_targets, [-1, 1.2])
        self.step({'w', 'e'})
        np.testing.assert_allclose(self.controller.frame_targets, [-.995, 1.195])
        for _ in range(500):
            self.step({'w', 'e'})
        np.testing.assert_allclose(self.controller.frame_targets, [1.2, -1])

    def test_return_to_normal_is_rate_limited_and_reentry_is_continuous(self):
        self.controller.handle_pressed({'c'})
        for _ in range(50):
            self.step({'w', 'e'})
        target = self.controller.frame_targets.copy()
        self.controller.handle_pressed({'c'})
        np.testing.assert_array_equal(self.controller.frame_targets, target)
        self.step({'q', 'r'})
        np.testing.assert_allclose(self.controller.frame_targets, [.245, -.245])
        self.controller.handle_pressed({'c'})
        self.step()
        np.testing.assert_allclose(self.controller.frame_targets, [.245, -.245])
        self.controller.handle_pressed({'c'})
        for _ in range(100):
            self.step(active=False)
        np.testing.assert_array_equal(self.controller.frame_targets, 0)

    def test_simulated_frames_track_nonzero_targets_and_return_to_zero(self):
        dt = self.model.opt.timestep

        def run(seconds, keys=()):
            for _ in range(round(seconds / dt)):
                self.step(keys, dt=dt)
                mujoco.mj_step(self.model, self.data)
            self.assertTrue(np.isfinite(self.data.qpos).all())

        run(2)
        self.controller.handle_pressed({'c'})
        run(.3, {'w', 'r'})
        run(5)
        frames = self.controller.qpos[self.controller.frames]
        np.testing.assert_allclose(self.controller.frame_targets, [.15, .15])
        np.testing.assert_allclose(self.data.qpos[frames], [.15, .15], atol=.025)
        self.controller.handle_pressed({'c'})
        run(5)
        np.testing.assert_array_equal(self.controller.frame_targets, 0)
        np.testing.assert_allclose(self.data.qpos[frames], 0, atol=.025)
        self.assertGreater(self.data.qpos[2], .07)


class MacLegKeysTests(unittest.TestCase):
    def test_leg_and_escape_physical_keys_and_single_mode_press(self):
        keyboard = MacTerminalKeyboard.__new__(MacTerminalKeyboard)
        keyboard.fd = 0
        keyboard.target = 'terminal'
        keyboard.previous = set()
        keyboard.CF = Mock()
        keyboard.CF.CFEqual.side_effect = lambda a, b: a == b
        keyboard.CG = Mock()
        down = {12, 13, 14, 15, 53, 8}
        keyboard.CG.CGEventSourceKeyState.side_effect = lambda state, code: code in down
        keyboard._focused_element = Mock(return_value='terminal')
        with patch('keyboard_macos.select.select', return_value=([], [], [])):
            expected = {'q', 'w', 'e', 'r', 'escape', 'c'}
            self.assertEqual(keyboard.poll()[:2], (expected, expected))
            self.assertEqual(keyboard.poll()[1], set())
            down.clear()
            self.assertEqual(keyboard.poll()[0], set())


if __name__ == '__main__':
    unittest.main()
