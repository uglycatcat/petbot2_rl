"""Synthetic boundary and acceptance tests; these are not simulation results."""

from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from step_demo.cli import main
from step_demo.config import DEFAULT_CONFIG, load_config
from step_demo.control import map_action
from step_demo.evaluation import CompletionTracker, EvaluationSample, Stage
from step_demo.model import read_limits
from step_demo.observations import MotionEstimate, ObservationBuilder, Proprioception, TerrainPacket


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.limits = read_limits(cls.config.model_dir)

    def sample(self, **changes):
        base = EvaluationSample(.1, .5, 0., 0., .1, 0., (0., 0.),
                                (True, True), True, False, False, False, False)
        return replace(base, **changes)

    def test_actions_enforce_source_limits_and_bilateral_identity(self):
        for hip in (-1., -.4, 0., .3, 1.):
            for wheel in (-1., -.2, 0., .8, 1.):
                targets = map_action((hip, wheel), self.limits, self.config.control)
                self.assertEqual(*targets.hip_positions_rad)
                self.assertEqual(*targets.rear_velocities_rad_s)
                self.assertTrue(-1 <= targets.hip_positions_rad[0] <= 1.2)
                self.assertTrue(0 <= targets.rear_velocities_rad_s[0] <= 25)
                self.assertEqual(targets.head_and_ears_rad, (0., 0., 0.))
        home = map_action((0., -1.), self.limits, self.config.control)
        self.assertEqual(home.hip_positions_rad, (0., 0.))
        self.assertEqual(home.rear_velocities_rad_s, (0., 0.))

    def test_nonfinite_or_out_of_bounds_actions_rejected(self):
        for action in ((float("nan"), 0), (0, float("inf")), (0, 1.1), (0,)):
            with self.assertRaises(ValueError):
                map_action(action, self.limits, self.config.control)

    def test_terrain_is_latched_once_and_changes_only_from_own_estimate(self):
        builder = ObservationBuilder()
        sensors = Proprioception((0., 0., -1.), (0., 0., 0.), (0., 0.), (0., 0.), (0., 0.))
        estimate = MotionEstimate(0., 0., 0., 0., 0.)
        with self.assertRaises(RuntimeError):
            builder.build(sensors, estimate, (0., -1.))
        packet = TerrainPacket(.02, .3, math.pi / 2)
        builder.receive_initial_terrain(packet)
        before = builder.build(sensors, estimate, (0., -1.))
        after = builder.build(sensors, replace(estimate, forward_m=.12), (0., -1.))
        self.assertEqual(len(before), 26)
        self.assertEqual(before[:4], after[:4])
        self.assertAlmostEqual(after[4], .18)
        with self.assertRaises(RuntimeError):
            builder.receive_initial_terrain(TerrainPacket(.02, .18, math.pi / 2))
        with self.assertRaises(TypeError):
            builder.build(sensors, estimate, (0., -1.), true_distance_m=.01)
        builder.reset()
        builder.receive_initial_terrain(packet)

    def test_invalid_observation_rejected(self):
        builder = ObservationBuilder()
        builder.receive_initial_terrain(TerrainPacket(.02, .3, math.pi / 2))
        sensors = Proprioception((0., 0., -1.), (0., 0., 0.), (0., 0.), (0., 0.), (0., 0.))
        with self.assertRaises(ValueError):
            builder.build(sensors, MotionEstimate(float("nan"), 0., 0., 0., 0.), (0., -1.))
        # A missing gyro value cannot be hidden by an extra hip value.
        malformed = replace(sensors, gyro_rad_s=(0., 0.), hip_position_rad=(0., 0., 0.))
        with self.assertRaises(ValueError):
            builder.build(malformed, MotionEstimate(0., 0., 0., 0., 0.), (0., -1.))

    def test_complete_requires_drive_after_alignment_and_continuous_stop(self):
        tracker = CompletionTracker(self.config.acceptance)
        self.assertEqual(tracker.update(self.sample()), Stage.PLATFORM_DRIVE)
        self.assertEqual(tracker.update(self.sample(forward_m=.66)), Stage.STOPPING)
        for _ in range(4):
            self.assertEqual(tracker.update(self.sample(forward_m=.66, speed_m_s=0.)), Stage.STOPPING)
        self.assertEqual(tracker.update(self.sample(forward_m=.66, speed_m_s=0.)), Stage.COMPLETE)

    def test_front_only_or_no_clearance_cannot_count_as_climb(self):
        for change in (dict(four_wheels_on_top=False), dict(front_clearance_verified=(False, True))):
            tracker = CompletionTracker(self.config.acceptance)
            for i in range(8):
                self.assertEqual(tracker.update(self.sample(forward_m=.5 + i * .1, speed_m_s=0., **change)), Stage.CROSSING)

    def test_stationary_on_platform_cannot_skip_extra_straight_distance(self):
        tracker = CompletionTracker(self.config.acceptance)
        for _ in range(10):
            self.assertEqual(tracker.update(self.sample(speed_m_s=0.)), Stage.PLATFORM_DRIVE)

    def test_path_pose_or_support_loss_clears_drive_credit(self):
        for change in (dict(lateral_m=.1), dict(heading_rad=.2),
                       dict(hip_error_rad=(0., .2)), dict(four_wheels_on_top=False)):
            tracker = CompletionTracker(self.config.acceptance)
            tracker.update(self.sample())
            tracker.update(self.sample(forward_m=.66))
            self.assertEqual(tracker.update(self.sample(forward_m=.66, **change)), Stage.CROSSING)
            self.assertEqual(tracker.update(self.sample(forward_m=.66, speed_m_s=0.)), Stage.PLATFORM_DRIVE)

    def test_collision_and_other_failures_latch_before_success(self):
        for flag in ("front_face_or_edge_hit", "rear_support_lost", "body_collision_or_fall", "deliberate_reverse"):
            tracker = CompletionTracker(self.config.acceptance)
            self.assertEqual(tracker.update(self.sample(**{flag: True})), Stage.FAILED)
            self.assertEqual(tracker.update(self.sample()), Stage.FAILED)

    def test_stop_timer_resets_when_robot_moves(self):
        tracker = CompletionTracker(self.config.acceptance)
        tracker.update(self.sample())
        for _ in range(4):
            tracker.update(self.sample(forward_m=.66, speed_m_s=0.))
        tracker.update(self.sample(forward_m=.66, yaw_rate_rad_s=.3))
        for _ in range(4):
            self.assertEqual(tracker.update(self.sample(forward_m=.66, speed_m_s=0.)), Stage.STOPPING)
        self.assertEqual(tracker.update(self.sample(forward_m=.66, speed_m_s=0.)), Stage.COMPLETE)

    def test_timeout_is_failure_and_reset_clears_episode_evidence(self):
        tracker = CompletionTracker(self.config.acceptance)
        self.assertEqual(tracker.update(self.sample(dt_s=15.)), Stage.FAILED)
        self.assertEqual(tracker.reason, "timeout")
        tracker.reset()
        self.assertEqual(tracker.front_clearance, [False, False])
        self.assertIsNone(tracker.drive_start_m)

    def test_changed_task_and_undocumented_initialization_rejected(self):
        for change in ("height", "parent", "nan"):
            data = json.loads(DEFAULT_CONFIG.read_text())
            if change == "height":
                data["terrain"]["height_m"] = .03
            elif change == "parent":
                data["initialize_from"] = "old_model.pt"
            else:
                data["acceptance"]["stop_hold_s"] = float("nan")
            with tempfile.TemporaryDirectory(prefix="petbot-step-test-") as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_unimplemented_training_and_evaluation_fail_explicitly(self):
        for command in ("train", "evaluate"):
            with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as caught:
                main([command])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("no run started", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
