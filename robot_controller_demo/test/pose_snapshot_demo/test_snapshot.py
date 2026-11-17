"""Exercise real MuJoCo state, sensor capture and disk reload without a window."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


if __package__:
    from . import main as demo_module
    from .pose_io import PoseLoader, SCENE, recorded_pose_path
else:
    import main as demo_module
    from pose_io import PoseLoader, SCENE, recorded_pose_path
PoseDemo = demo_module.PoseDemo
import mujoco


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.demo = PoseDemo()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_default_pose_hold_under_real_dynamics(self):
        np.testing.assert_array_equal(self.demo.data.qpos, self.demo.model.qpos0)
        for _ in range(1000):
            self.demo.step()
        self.demo.refresh()
        self.assertTrue(np.isfinite(self.demo.data.qpos).all())
        self.assertLess(self.demo.imu()["tilt_deg"], 2.0)
        self.assertLess(np.max(np.abs(self.demo.data.qpos[self.demo.qadr] - self.demo.home)), 0.02)
        for name in ("left_rear_wheel", "right_rear_wheel"):
            self.assertEqual(self.demo.data.ctrl[self.demo.model.actuator(name).id], 0.0)

    def test_fallen_pose_and_imu_reload_into_fresh_model(self):
        # Nontrivial yaw and a >90 degree roll; horizontal drift is not a pose.
        self.demo.data.qpos[:3] = [0.4, -0.7, 0.22]
        self.demo.data.qpos[3:7] = np.array([0.3, 0.8, -0.2, 0.4]) / np.sqrt(0.93)
        self.demo.data.qpos[self.demo.qadr] = [0.03, -0.02, 0.01, -0.04, 0.05]
        self.demo.data.qvel[:] = np.linspace(-0.2, 0.2, self.demo.model.nv)
        saved_qpos = self.demo.data.qpos.copy()
        saved_qvel = self.demo.data.qvel.copy()
        path = self.demo.save(self.directory)
        record = json.loads(path.read_text())
        np.testing.assert_array_equal(self.demo.data.qpos, saved_qpos)
        self.assertNotIn("root_position_m", record)
        self.assertEqual(record["format"], demo_module.FORMAT)
        self.assertGreater(record["imu"]["tilt_deg"], 90)
        np.testing.assert_array_equal(record["qvel_at_capture"], saved_qvel)
        np.testing.assert_allclose(record["imu"]["gyro_rad_s"], self.demo.data.sensor("imu_gyro").data)
        np.testing.assert_allclose(record["imu"]["accel_m_s2"], self.demo.data.sensor("imu_accel").data)
        loaded = PoseDemo()
        loaded.data.xfrc_applied[:] = 12
        loaded.data.qfrc_applied[:] = 4
        loaded.data.qvel[:] = 3
        placement = loaded.load(path)
        expected = saved_qpos.copy()
        expected[:2] = self.demo.model.qpos0[:2]
        np.testing.assert_array_equal(loaded.data.qpos, expected)
        self.assertEqual(placement["height_adjustment_m"], 0)
        np.testing.assert_array_equal(loaded.data.qvel, 0)
        np.testing.assert_array_equal(loaded.data.xfrc_applied, 0)
        np.testing.assert_array_equal(loaded.data.qfrc_applied, 0)
        np.testing.assert_allclose(loaded.imu()["quat_wxyz"], record["imu"]["quat_wxyz"], atol=1e-14)
        self.assertEqual(loaded.data.time, 0)
        loaded.reset()
        np.testing.assert_array_equal(loaded.data.qpos, loaded.model.qpos0)

    def test_legacy_pose_is_centered_and_lifted_without_modifying_file(self):
        record = self.demo.capture()
        record["format"] = demo_module.LEGACY_FORMAT
        qpos = self.demo.model.qpos0.copy()
        qpos[:3] = [25.0, -18.0, -0.2]
        qpos[3:7] = [0.5, np.sqrt(0.75), 0, 0]
        qpos[self.demo.qadr] = [0.01, -0.02, 0.03, -0.04, 0.05]
        record["qpos"] = qpos.tolist()
        record["root_position_m"] = qpos[:3].tolist()
        path = self.directory / "old_pose.json"
        original = json.dumps(record).encode()
        path.write_bytes(original)
        placement = self.demo.load(path)
        np.testing.assert_array_equal(self.demo.data.qpos[:2], self.demo.model.qpos0[:2])
        np.testing.assert_array_equal(self.demo.data.qpos[3:], qpos[3:])
        self.assertGreater(placement["height_adjustment_m"], 0.2)
        self.assertGreaterEqual(self.demo.floor_clearance(), demo_module.FLOOR_CLEARANCE_M - 1e-9)
        self.assertEqual(path.read_bytes(), original)
        # Re-saving and reloading a corrected pose must not keep raising it.
        expected = self.demo.data.qpos.copy()
        new_path = self.demo.save(self.directory)
        self.demo.load(new_path)
        np.testing.assert_array_equal(self.demo.data.qpos, expected)

    def test_new_pose_record_ignores_horizontal_drag_distance(self):
        self.demo.data.qpos[2] = 0.3
        self.demo.data.qpos[:2] = [0.3, -0.7]
        first = self.demo.capture()
        self.demo.data.qpos[:2] = [200.0, -180.0]
        second = self.demo.capture()
        self.assertEqual(first["qpos"], second["qpos"])
        self.assertEqual(first["root_height_m"], second["root_height_m"])
        np.testing.assert_array_equal(self.demo.data.qpos[:2], [200.0, -180.0])

    def test_failed_floor_check_keeps_live_pose(self):
        path = self.demo.save(self.directory)
        self.demo.data.qpos[0] = 1.25
        expected = self.demo.data.qpos.copy()
        with patch.object(self.demo, "floor_clearance", return_value=float("nan")):
            with self.assertRaises(ValueError):
                self.demo.load(path)
        np.testing.assert_array_equal(self.demo.data.qpos, expected)

    def test_capture_refreshes_imu_after_integration(self):
        self.demo.data.qpos[2] = 0.5
        self.demo.data.qvel[3:6] = [1.0, 0.4, -0.3]
        self.demo.step()
        record = self.demo.capture()
        quat = np.array(record["imu"]["quat_wxyz"])
        base_quat = np.array(record["root_quat_wxyz"])
        self.assertAlmostEqual(abs(np.dot(quat, base_quat)), 1.0, places=12)

    def test_invalid_snapshot_does_not_change_current_pose(self):
        baseline = self.demo.capture()
        broken = []
        bad = copy.deepcopy(baseline)
        bad["model"]["mjcf_sha256"] = "different"
        broken.append(bad)
        for value in ([0.0], [float("nan")] * self.demo.model.nq, [0.0] * self.demo.model.nq):
            bad = copy.deepcopy(baseline)
            bad["qpos"] = value
            broken.append(bad)
        self.demo.data.qpos[0] = 1.25
        expected = self.demo.data.qpos.copy()
        for record in broken:
            with self.subTest(record=record):
                path = self.directory / "invalid.json"
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    self.demo.load(path)
                np.testing.assert_array_equal(self.demo.data.qpos, expected)

    def test_multiple_saves_preserve_files_and_latest(self):
        with self.assertRaises(FileNotFoundError):
            demo_module.latest_snapshot(self.directory)
        first = self.demo.save(self.directory)
        self.demo.data.qpos[0] = 0.3
        second = self.demo.save(self.directory)
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_file())
        self.assertEqual(demo_module.latest_snapshot(self.directory), second)
        self.assertEqual(len(list(self.directory.iterdir())), 2)

    def test_recorded_library_loads_into_caller_owned_simulation(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        data = mujoco.MjData(model)
        loader = PoseLoader(model=model, data=data)
        for number in range(1, 5):
            with self.subTest(pose=number):
                path = recorded_pose_path(f"pose_{number:02d}")
                original = path.read_bytes()
                record = json.loads(original)
                data.qvel[:] = 5
                data.ctrl[:] = 0.1
                data.xfrc_applied[:] = 2
                data.qfrc_applied[:] = 3
                data.time = 12.0
                placement = loader.load(path)
                self.assertIs(loader.model, model)
                self.assertIs(loader.data, data)
                np.testing.assert_array_equal(data.qpos[:2], model.qpos0[:2])
                np.testing.assert_array_equal(data.qpos[3:], record["qpos"][3:])
                np.testing.assert_array_equal(data.qvel, 0)
                np.testing.assert_array_equal(data.ctrl, 0)
                np.testing.assert_array_equal(data.xfrc_applied, 0)
                np.testing.assert_array_equal(data.qfrc_applied, 0)
                self.assertEqual(data.time, 0)
                self.assertGreaterEqual(placement["floor_clearance_m"], demo_module.FLOOR_CLEARANCE_M - 1e-9)
                self.assertEqual(path.read_bytes(), original)

    def test_recorded_pose_id_must_exist(self):
        with self.assertRaisesRegex(ValueError, "未知姿态"):
            recorded_pose_path("pose_99")


if __name__ == "__main__":
    unittest.main()
