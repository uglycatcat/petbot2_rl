"""Continuous targets, symmetric boundary exposure, RNG replay and probe isolation."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config
from upright.runtime import Runtime
from upright.controller import Phase
from test_contracts import MODEL, state

CFG = load_config()

class BoundaryCommands(unittest.TestCase):
    def test_boundary_mixture_retains_continuous_range_and_both_directions(self):
        p = Runtime(4096, 'cpu', .06, CFG, MODEL); p.level[:] = CFG.max_course_level
        cmd = p.training_command(torch.arange(4096))
        limits = torch.tensor(CFG.motion_stages()[-1][:2])
        self.assertTrue((cmd.abs() <= limits).all())
        for axis in range(2):
            edge = cmd[:, axis].abs() == limits[axis]
            interior = (cmd[:, axis].abs() > 0) & ~edge
            self.assertTrue(.11 < float(edge.float().mean()) < .19)
            self.assertGreater(int(interior.sum()), 1000)
            self.assertGreater(int((cmd[:, axis] == limits[axis]).sum()), 200)
            self.assertGreater(int((cmd[:, axis] == -limits[axis]).sum()), 200)
            self.assertLess(abs(float(cmd[:, axis].mean())), .025)
        self.assertTrue(((cmd[:, 0] == 0) & (cmd[:, 1] == 0)).any())
        self.assertTrue(((cmd[:, 0] == 0) & (cmd[:, 1] != 0)).any())
        self.assertTrue(((cmd[:, 0] != 0) & (cmd[:, 1] == 0)).any())

    def test_rng_replay_and_subset_preserve_unselected_commands(self):
        p = Runtime(32, 'cpu', .06, CFG, MODEL); p.level[:] = 6
        ids = torch.tensor([1, 4, 9, 17]); rng = p.generator.get_state().clone()
        first = p.training_command(ids); p.generator.set_state(rng)
        torch.testing.assert_close(first, p.training_command(ids), rtol=0, atol=0)
        before = p.control.command.clone(); p.control.set_command(first, ids=ids)
        mask = torch.ones(32, dtype=torch.bool); mask[ids] = False
        torch.testing.assert_close(p.control.command[mask], before[mask])

    def test_low_course_and_disabled_feature_preserve_original_draws(self):
        for level, fraction in [(2, .25), (6, 0.)]:
            cfg = replace(CFG, boundary_command_fraction=fraction)
            p = Runtime(64, 'cpu', .06, cfg, MODEL); p.level[:] = level
            before = p.generator.get_state().clone()
            old = (2*p.uniform(64, 2)-1)*p.stage_ranges[p.level, :2]
            category = (p.uniform(64)[:, 0]*5).long()
            old[category == 0] = 0; old[category == 1, 1] = 0; old[category == 2, 0] = 0
            after = p.generator.get_state().clone(); p.generator.set_state(before)
            torch.testing.assert_close(old, p.training_command(torch.arange(64)), rtol=0, atol=0)
            torch.testing.assert_close(after, p.generator.get_state())

    def test_independent_probe_keeps_original_requests_and_rng(self):
        runtimes = [Runtime(32, 'cpu', .06, replace(CFG, boundary_command_fraction=f), MODEL) for f in (0., .25)]
        for p in runtimes:
            p.probe_level = 6; p.probe_duration = 60.; p.reset(torch.arange(32))
            p.control.phase[:] = Phase.BALANCE
            before = p.generator.get_state().clone()
            for t in (3.1, 6.2, 9.3):
                p.balance_elapsed[:] = t; p.command_age[:] = 3.1; p.before_action(state(32))
            torch.testing.assert_close(before, p.generator.get_state())
        torch.testing.assert_close(runtimes[0].control.command, runtimes[1].control.command, rtol=0, atol=0)
        torch.testing.assert_close(runtimes[0].control.motion_reference, runtimes[1].control.motion_reference, rtol=0, atol=0)

    def test_invalid_fractions_rejected(self):
        for value in (-.01, 1., float('nan'), float('inf')):
            with self.assertRaises(ValueError): replace(CFG, boundary_command_fraction=value).validate()

if __name__ == '__main__': unittest.main()
