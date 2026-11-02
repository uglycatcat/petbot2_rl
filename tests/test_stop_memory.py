"""Boundary and information-source checks for the deployable stopping contract."""
import json
import unittest
from pathlib import Path
import torch
from step_demo.stop_memory import SensorStopMemory, wheel_request
from step_demo.runtime import load_training
from step_demo.rewards import settled_progress, component_increment


class StopMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = SensorStopMemory(2, 'cpu')
        self.packet = torch.tensor([[.02, .30, torch.pi/2]]*2)
        self.hips = torch.zeros(2, 2)

    def step(self, x=.6, rear=.4, hips=None):
        com = torch.tensor([[x, 0., 0.]]*2)
        axle = torch.tensor([[rear, 0., 0.]]*2)
        self.memory.update(self.packet, com, axle, self.hips if hips is None else hips, .02)

    def test_ground_approach_and_unrestored_hips_cannot_start_drive_memory(self):
        for _ in range(20):self.step(rear=.30)
        self.assertFalse(self.memory.started.any())
        for _ in range(20):self.step(hips=torch.full((2, 2), .4))
        self.assertFalse(self.memory.started.any())

    def test_confirmation_and_short_drive_precede_latched_stop_intent(self):
        self.step();self.step();self.assertFalse(self.memory.started.any())
        self.step();self.assertTrue(self.memory.started.all())
        self.step(x=.74);self.assertFalse(self.memory.stop_intent.any())
        self.step(x=.76);self.assertTrue(self.memory.stop_intent.all())
        self.step(x=.5, rear=.2, hips=torch.ones(2, 2))
        self.assertTrue(self.memory.stop_intent.all())
        self.assertTrue(torch.isfinite(self.memory.observation()).all())

    def test_short_home_excursion_resets_confirmation_before_entry(self):
        self.step();self.step();self.step(hips=torch.ones(2, 2));self.step()
        self.assertFalse(self.memory.started.any())

    def test_reset_is_per_environment_and_drops_all_task_memory(self):
        for _ in range(3):self.step()
        self.step(x=.8);self.memory.reset(torch.tensor([0]))
        torch.testing.assert_close(self.memory.observation()[0], torch.zeros(3))
        torch.testing.assert_close(self.memory.observation()[1], torch.ones(3))
        self.assertEqual(float(self.memory.anchor[0]), 0.)

    def test_initial_edge_distance_changes_entry_without_any_truth_inputs(self):
        self.packet[1, 1] = .6
        for _ in range(3):self.step()
        self.assertEqual(self.memory.started.tolist(), [True, False])

    def test_reverse_motion_cannot_accumulate_short_drive_distance(self):
        for _ in range(3):self.step()
        for x in [.5, .6, .5, .6, .5, .6]:self.step(x=x)
        self.assertFalse(self.memory.stop_intent.any())
        torch.testing.assert_close(self.memory.fraction, torch.zeros(2))

    def test_same_instantaneous_pose_preserves_distinct_task_history(self):
        for _ in range(3):self.step()
        self.step(x=.8);self.memory.reset(torch.tensor([0]));self.step(x=.6)
        self.assertFalse(torch.equal(self.memory.observation()[0], self.memory.observation()[1]))

    def test_zero_is_exact_and_available_over_half_the_action_domain(self):
        a = torch.tensor([-1., -.5, -.001, 0., .1, 1.])
        torch.testing.assert_close(wheel_request(a, 'zero_centered'), torch.tensor([0., 0., 0., 0., 2.5, 25.]))

    def test_same_physical_requests_can_be_compared_across_parameterizations(self):
        a = torch.linspace(-1., 1., 10001)
        self.assertTrue(torch.equal(wheel_request(a), wheel_request((a+1)*.5, 'zero_centered')))

    def test_braking_credit_can_continue_through_support_gaps_but_is_bounded(self):
        previous = torch.zeros(1, 8)
        total = 0.
        for speed in [.6, .3, .1, 0., 0., .4, 0.]:
            parts = settled_progress(torch.zeros(1, 8), torch.zeros(1), torch.zeros(1),
                torch.zeros(1), torch.tensor([False]), torch.tensor([True]), torch.tensor([speed]), True, 4.)
            gain, previous = component_increment(previous, parts, torch.tensor([False]))
            total += float(gain)
        self.assertAlmostEqual(total, 4., places=6)
        off_platform = settled_progress(torch.zeros(1, 8), torch.zeros(1), torch.zeros(1),
            torch.zeros(1), torch.tensor([False]), torch.tensor([False]), torch.zeros(1), True, 4.)
        self.assertEqual(float(off_platform[0, 7]), 0.)

    def test_formal_configs_keep_original_physics_acceptance_and_scratch_budget(self):
        root = Path(__file__).resolve().parents[1]/'configs'
        for suffix in ('seed70', 'seed71', 'gate', 'scale_gate'):
            old = load_training(root/f'round15_{suffix}.json')
            new = load_training(root/f'round16_{suffix}.json')
            self.assertEqual(new.pop('task_memory'), 'sensor_stop_v16')
            self.assertEqual(new.pop('wheel_mapping'), 'zero_centered')
            self.assertEqual(new.pop('reward_version'), 'stop_v16')
            old.pop('reward_version');self.assertEqual(old, new)


if __name__ == '__main__':unittest.main()
