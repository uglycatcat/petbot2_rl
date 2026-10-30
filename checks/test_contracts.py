"""CPU contract tests: no simulator, PPO rollout, GPU allocation or historical models."""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest
import torch
from upright.config import Config, require_execution, require_gpu_isolation
from upright.controller import Controller, Phase, Reason, wheel_mix
from upright.curriculum import Curriculum
from upright.geometry import WheelGeometry, orientation
from upright.model import load_model, wheel_geometry, source_fingerprint
from upright.runtime import Runtime, OBSERVATION_DIM
from upright.experiment import SCHEMA, check_identity, validate_resume, atomic_json

CFG = Config(num_envs=32)
MODEL = load_model()


def state(n=2):
    return dict(g=torch.tensor([[0., 0., -1.]]).repeat(n, 1), yaw=torch.zeros(n),
                gyro=torch.zeros(n, 3), motion=torch.zeros(n, 2), q=torch.zeros(n, 5),
                qd=torch.zeros(n, 5), wheel=torch.zeros(n, 2), gap=torch.zeros(n, 2), rear_skew=torch.zeros(n))


def truth(n=2, raised=False):
    return dict(front_z=torch.full((n, 2), .15 if raised else .045), rear_z=torch.full((n, 2), .031),
                body_contact=torch.zeros(n, dtype=torch.bool), root_height=torch.full((n,), .085))


def controller(n=2):
    return Controller(n, 'cpu', .06, CFG, MODEL, .10)


class ControlContracts(unittest.TestCase):
    def test_wheel_limit_and_mirror(self):
        common, differential = torch.tensor([25., -25., 5.]), torch.tensor([8., 8., -2.])
        a = wheel_mix(common, differential, 25)
        b = wheel_mix(common, -differential, 25)
        torch.testing.assert_close(a, b.flip(-1))
        self.assertLessEqual(float(a.abs().max()), 25)

    def test_ordinary_limits_independent_of_course(self):
        c, s = controller(), state()
        c.set_command([[.75, 0], [-.75, 0]], hip=torch.tensor([1.1, -.9]))
        target = c.targets(torch.zeros(2, 3), s)
        torch.testing.assert_close(target[:, 1:], torch.tensor([[25., 25.], [-25., -25.]]))
        torch.testing.assert_close(target[:, 0], torch.tensor([1.1, -.9]))

    def test_space_preserves_executed_targets_and_real_motion(self):
        c, s = controller(), state()
        c.executed[:] = torch.tensor([.2, 10., 10.]); s['motion'][:, 0] = .3
        c.request_rise(s, torch.arange(2))
        torch.testing.assert_close(c.executed, torch.tensor([[.2, 10., 10.]]).repeat(2, 1))
        torch.testing.assert_close(c.transition_motion[:, 0], torch.full((2,), .3))

    def test_rise_ignores_policy_turn_channel(self):
        c, s = controller(), state()
        c.request_rise(s, torch.arange(2))
        a = torch.tensor([[0., .5, 1.], [0., .5, -1.]])
        c.targets(a, s)
        torch.testing.assert_close(c.executed[0], c.executed[1])

    def test_not_yet_lifted_is_not_a_front_return(self):
        c, s = controller(), state(); c.request_rise(s, torch.arange(2))
        for _ in range(10):
            m = c.update(s)
        self.assertFalse(m['failure'].any())
        self.assertTrue((c.phase == Phase.RISE).all())

    def test_unexpected_return_exits_and_cannot_auto_rise(self):
        c, s = controller(), state(); c.request_rise(s, torch.arange(2))
        s['gap'][:] = .04
        c.update(s)
        s['gap'][:] = 0
        for _ in range(4): c.update(s)
        self.assertTrue((c.phase == Phase.EXIT).all())
        self.assertTrue((c.reason == Reason.FRONT_RETURN).all())
        s['gap'][:] = .1
        for _ in range(20): c.update(s)
        self.assertTrue((c.phase == Phase.EXIT).all())
        c.request_rise(s, torch.arange(2))
        self.assertTrue((c.phase == Phase.RISE).all())

    def test_landing_handoff_does_not_wait_for_hip_zero(self):
        c, s = controller(), state(); c.phase[:] = Phase.BALANCE
        c.executed[:] = torch.tensor([.5, 10., 10.]); s['q'][:, 3:] = .5; s['motion'][:, 0] = .3
        c.request_land(s, torch.arange(2))
        for _ in range(4): c.update(s)
        self.assertTrue((c.phase == Phase.HANDOFF).all())
        torch.testing.assert_close(c.command[:, 0], torch.full((2,), .3))
        self.assertFalse(c.completed.any())
        # First handoff output is continuous; ordinary controller now owns the hip return.
        c.age[:] = 0
        targets = c.targets(torch.ones(2, 3), s)
        torch.testing.assert_close(targets[:, 1:], torch.full((2, 2), 10.))
        self.assertTrue((targets[:, 0] < .5).all())
        s['q'][:, 3:] = 0
        for _ in range(12): c.update(s)
        self.assertTrue(c.completed.all())

    def test_fall_has_priority_over_completion(self):
        c, s = controller(), state(); c.phase[:] = Phase.HANDOFF
        c.handoff_run[:] = CFG.handoff_hold_s
        s['g'][:, 2] = 1
        m = c.update(s)
        self.assertTrue(m['failure'].all()); self.assertFalse(m['success'].any())
        self.assertTrue((c.phase == Phase.EXIT).all())

    def test_two_wheel_hip_is_policy_owned(self):
        c, s = controller(), state(); c.phase[:] = Phase.BALANCE
        c.set_command([[0, 0], [0, 0]], hip=torch.tensor([1.2, -1.]))
        c.targets(torch.zeros(2, 3), s)
        torch.testing.assert_close(c.executed[:, 0], torch.full((2,), .1))

    def test_heading_counteracts_drift(self):
        c, s = controller(), state(); c.request_rise(s, torch.arange(2))
        s['yaw'][:] = torch.tensor([.3, -.3])
        a = c.targets(torch.zeros(2, 3), s)
        torch.testing.assert_close(a[0, 1:], a[1, 1:].flip(-1))
        self.assertGreater(float(a[0, 1]), float(a[0, 2]))


class LearningContracts(unittest.TestCase):
    def test_moving_probe_reaches_both_hip_initial_states(self):
        p = Runtime(32, 'cpu', .06, CFG, MODEL)
        p.probe_level = 2
        with torch.inference_mode():
            p.reset(torch.arange(32))
        torch.testing.assert_close(p.initial_hip[:6], torch.full((6,), -CFG.initial_hip))
        torch.testing.assert_close(p.initial_hip[12:18], torch.full((6,), CFG.initial_hip))
        torch.testing.assert_close(p.control.ordinary_hip, p.initial_hip)
        p.probe_level = 0
        p.reset(torch.arange(32))
        self.assertFalse(p.initial_hip.any())

    def test_probe_reset_can_leave_inference_mode(self):
        p, s = Runtime(2, 'cpu', .06, CFG, MODEL), state()
        p.reset(torch.arange(2)); p.control.request_rise(s, torch.arange(2)); p.invoked[:] = True
        s['gap'][:] = .1
        with torch.inference_mode():
            for _ in range(8): p.advance(s, truth(raised=True))
        p.reset(torch.arange(2))
        self.assertFalse(p.control.entered.any())

    def test_idle_rise_has_no_positive_stream(self):
        p, s = Runtime(2, 'cpu', .06, CFG, MODEL), state()
        p.reset(torch.arange(2)); p.control.request_rise(s, torch.arange(2)); p.invoked[:] = True
        rewards = [p.advance(s, truth())['reward'] for _ in range(30)]
        self.assertLessEqual(float(torch.stack(rewards).max()), 0)

    def test_geometry_cannot_claim_real_support_by_itself(self):
        p, s = Runtime(2, 'cpu', .06, CFG, MODEL), state()
        p.reset(torch.arange(2)); p.control.request_rise(s, torch.arange(2)); p.invoked[:] = True
        s['gap'][:] = .1
        for _ in range(45): m = p.advance(s, truth())
        self.assertFalse(m['goal'].any()); self.assertFalse(m['complete'].any())
        self.assertFalse(p.balance_seen.any())

    def test_rise_goal_is_separate_from_complete_cycle(self):
        p, s = Runtime(2, 'cpu', .06, CFG, MODEL), state()
        p.reset(torch.arange(2)); p.control.request_rise(s, torch.arange(2)); p.invoked[:] = True
        s['gap'][:] = .1
        for _ in range(40): m = p.advance(s, truth(raised=True))
        self.assertTrue(m['goal'].all()); self.assertFalse(m['complete'].any())

    def test_observations_exclude_truth_and_global_yaw(self):
        p, s = Runtime(2, 'cpu', .06, CFG, MODEL), state()
        p.reset(torch.arange(2)); p.control.phase[:] = Phase.BALANCE
        a = p.observation(s)
        s['yaw'][:] = 2.5
        torch.testing.assert_close(a, p.observation(s))
        self.assertEqual(a.shape, (2, OBSERVATION_DIM))

    def test_curriculum_never_advances_on_iterations_alone(self):
        c = Curriculum(CFG)
        for update in (100, 200, 1000): c.observe_probe(update, 0, 32, 0)
        self.assertEqual(c.level, 0)
        c.observe_probe(1100, 32, 32, 0)
        self.assertEqual(c.level, 0)
        c.observe_probe(1200, 32, 32, 0)
        self.assertEqual(c.level, 1)
        with self.assertRaises(ValueError): c.observe_probe(1200, 32, 32, 1)
        restored = Curriculum(CFG); restored.load_state_dict(c.state_dict())
        self.assertEqual(restored.state_dict(), c.state_dict())

    def test_continuous_signed_command_sampling_and_stop(self):
        cfg = replace(CFG, num_envs=512)
        p, s = Runtime(512, 'cpu', .06, cfg, MODEL), state(512)
        p.course.level = 2; p.reset(torch.arange(512))
        p.level[:] = 2; p.control.phase[:] = Phase.BALANCE; p.command_age[:] = 2
        p.before_action(s)
        commands = p.control.command
        self.assertTrue((commands[:, 0] > 0).any() and (commands[:, 0] < 0).any())
        self.assertTrue((commands[:, 1] > 0).any() and (commands[:, 1] < 0).any())
        self.assertGreater(len(commands[:, 0].unique()), 20)
        self.assertTrue((commands == 0).all(-1).any())

    def test_delivery_budget_can_exceed_demo_budget(self):
        replace(CFG, iterations=10000).validate()

    def test_execution_stays_disabled(self):
        with patch('upright.config.json.loads', return_value={}):
            for kind in ('training', 'engineering', 'evaluation'):
                with self.assertRaises(RuntimeError): require_execution(kind)

    def test_gpu0_and_gpu1_are_rejected(self):
        for gpu in ('0', '1'):
            with patch.dict('os.environ', {'PETBOT_HOST_GPU': gpu, 'PETBOT_GPU_UUID': 'GPU-test'}):
                with self.assertRaises(RuntimeError): require_gpu_isolation()


class GeometryAndRecovery(unittest.TestCase):
    def test_default_four_wheel_geometry(self):
        geo = WheelGeometry(wheel_geometry(), 'cpu')
        gap, skew = geo.estimate(torch.tensor([[0., 0., -1.]]), torch.zeros(1, 2))
        torch.testing.assert_close(gap, torch.zeros_like(gap)); torch.testing.assert_close(skew, torch.zeros_like(skew))
        self.assertAlmostEqual(geo.track, .1, places=5)

    def test_local_motion_rotation(self):
        q = torch.tensor([[0., 0., 0., 1.], [0., 0., 2**-.5, 2**-.5]])
        g, _, gyro, motion = orientation(q, torch.tensor([[.2, 0, 0], [0, .2, 0]]), torch.tensor([[0., 0., .3]]).repeat(2, 1))
        torch.testing.assert_close(motion[0], motion[1]); torch.testing.assert_close(g[0], g[1])

    def test_resume_rejects_cross_run_or_changed_code(self):
        current = dict(schema=SCHEMA, initialization='random', run_id='a', code_sha256='c', model_sha256='m', config_sha256='f')
        check_identity(current, current)
        for key in ('run_id', 'code_sha256', 'model_sha256', 'config_sha256', 'schema', 'initialization'):
            with self.assertRaises(ValueError): check_identity(dict(current, **{key: 'other'}), current)

    def test_resume_rejects_completed_and_old_checkpoints(self):
        identity = dict(schema=SCHEMA, initialization='random', run_id='a', code_sha256='c', model_sha256='m', config_sha256='f')
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); (folder/'checkpoints').mkdir()
            atomic_json(folder/'run.json', identity)
            atomic_json(folder/'status.json', dict(state='complete_execution_only', run_id='a'))
            atomic_json(folder/'checkpoints/latest.json', dict(file='update_000100.pt'))
            with self.assertRaises(ValueError): validate_resume(folder, folder/'checkpoints/update_000100.pt', identity)
            atomic_json(folder/'status.json', dict(state='interrupted', run_id='a'))
            with self.assertRaises(ValueError): validate_resume(folder, folder/'checkpoints/update_000050.pt', identity)
            self.assertEqual(validate_resume(folder, folder/'checkpoints/update_000100.pt', identity)['run_id'], 'a')

    def test_source_limits_unchanged(self):
        self.assertEqual(MODEL['actuators']['left_rear_wheel']['control_range'], [-25, 25])
        self.assertEqual(MODEL['actuators']['right_rear_wheel']['effort'], .38)
        self.assertEqual(len(source_fingerprint()), 64)


if __name__ == '__main__':
    unittest.main(verbosity=2)
