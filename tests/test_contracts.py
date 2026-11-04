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
from step_demo.model import read_limits
from step_demo.observations import MotionEstimate, ObservationBuilder, Proprioception, TerrainPacket


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.limits = read_limits(cls.config.model_dir)


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

    def test_training_rejects_foreign_initialization(self):
        from step_demo.config import PROJECT_ROOT
        from step_demo.runtime import load_training
        data = json.loads((PROJECT_ROOT/'configs/train.json').read_text())
        data['pretrained_models']=['historical.pt']
        with tempfile.TemporaryDirectory(prefix='step-config-') as directory:
            path=Path(directory)/'train.json';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): load_training(path)


if __name__ == "__main__":
    unittest.main()
