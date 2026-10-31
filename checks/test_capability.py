"""Capability progression must preserve provenance, low-range replay and entry independence."""
from dataclasses import replace
from pathlib import Path
import unittest
import torch
from upright.config import load_config
from upright.curriculum import Curriculum
from upright.runtime import Runtime
from upright.controller import Phase
from test_contracts import MODEL, state

CFG = load_config(Path(__file__).resolve().parents[1]/'configs/round05.json')

class CapabilityContracts(unittest.TestCase):
    def test_two_passes_per_stage_and_no_untrained_final_promotion(self):
        c = Curriculum(CFG)
        for u in (50, 100): c.observe_probe(u, 32, 32, 0)
        self.assertEqual(c.level, 1)
        for u in (150, 200): c.observe_probe(u, 32, 32, 1)
        self.assertEqual(c.level, 2)
        c.observe_probe(300, 25, 32, 2)
        c.observe_probe(400, 32, 32, 2)
        self.assertEqual(c.level, 2)
        c.observe_probe(500, 32, 32, 2)
        self.assertEqual(c.level, 3)
        c.observe_probe(900, 32, 32, 3)
        r = c.observe_probe(1000, 32, 32, 3)
        self.assertEqual(c.level, 3)
        self.assertTrue(r['promotion_deferred_at_budget'])

    def test_pending_probe_and_level_restore(self):
        c = Curriculum(CFG)
        self.assertEqual(c.next_probe_update(0), 50)
        self.assertTrue(c.probe_due(50))
        c.observe_probe(50, 32, 32, 0)
        saved = c.state_dict(); restored = Curriculum(CFG); restored.load_state_dict(saved)
        self.assertFalse(restored.probe_due(50))
        self.assertTrue(restored.probe_due(100))
        restored.level = 3
        self.assertEqual(restored.next_probe_update(250), 300)
        self.assertFalse(restored.probe_due(250))
        restored.load_state_dict(dict(saved, level=CFG.max_course_level))
        self.assertEqual(restored.level, 7)

    def test_entry_ranges_independent_of_motion_and_mirrored(self):
        p = Runtime(32, 'cpu', .06, CFG, MODEL)
        p.probe_level = 3; p.reset(torch.arange(32))
        self.assertAlmostEqual(float(p.initial_hip.abs().max()), .1, places=6)
        self.assertAlmostEqual(float(p.control.command[:, 0].abs().max()), .1, places=6)
        p.control.phase[:] = Phase.BALANCE; p.command_age[:] = 2; p.balance_elapsed[:] = 2.1
        p.before_action(state(32))
        torch.testing.assert_close(p.control.command[:, 0], torch.full((32,), .2))
        p.command_age[:] = 2; p.balance_elapsed[:] = 10.1; p.before_action(state(32))
        torch.testing.assert_close(p.control.command[::2, 1], -p.control.command[1::2, 1])
        self.assertAlmostEqual(float(p.control.command[:, 1].abs().max()), .5, places=6)
        p.probe_level = 4; p.reset(torch.arange(32))
        self.assertAlmostEqual(float(p.initial_hip.abs().max()), .2, places=6)
        self.assertAlmostEqual(float(p.control.command[:, 0].abs().max()), .1, places=6)

    def test_replay_preserves_static_and_all_earlier_moving_stages(self):
        p = Runtime(2048, 'cpu', .06, CFG, MODEL)
        p.course.level = 5; p.reset(torch.arange(2048))
        self.assertEqual(set(p.level.tolist()), {1,2,3,4,5})
        fraction = float((p.level < 5).float().mean())
        self.assertTrue(.15 < fraction < .25)
        s = state(2048); p.control.phase[:] = Phase.BALANCE; p.command_age[:] = 2
        p.before_action(s)
        self.assertTrue((p.control.command.abs() <= p.stage_ranges[p.level, :2]+1e-6).all())
        self.assertGreater(len(p.control.command[:, 0].unique()), 100)
        self.assertTrue((p.control.command[:, 0] < 0).any())

    def test_invalid_ladders_are_rejected(self):
        for stages in ([[.1,.3,.1,.1],[.76,.5,.1,.1]], [[.1,.3,.1,.1],[.2,float('nan'),.1,.1]],
                       [[.2,.3,.1,.1]], [[.1,.3,.1,.1],[.05,.3,.1,.1]]):
            with self.assertRaises(ValueError): replace(CFG, capability_stages=stages).validate()
        with self.assertRaises(ValueError): replace(CFG, bootstrap_probe_interval=30).validate()

    def test_finer_course_preserves_full_speed_mirrors_and_entry_ranges(self):
        cfg = load_config()
        p = Runtime(32, 'cpu', .06, cfg, MODEL)
        for level, speed in ((7, .6), (8, .75)):
            p.probe_level = level; p.reset(torch.arange(32))
            self.assertAlmostEqual(float(p.initial_hip.abs().max()), .2, places=6)
            self.assertAlmostEqual(float(p.control.command[:, 0].abs().max()), .3, places=6)
            dwell = float(p.stage_dwell[level])
            for index, sign in ((1, 1), (2, -1)):
                p.control.phase[:] = Phase.BALANCE
                p.command_age[:] = dwell+.1; p.balance_elapsed[:] = index*dwell+.1
                p.before_action(state(32))
                torch.testing.assert_close(p.control.command[:, 0], torch.full((32,), sign*speed))
            p.command_age[:] = dwell+.1; p.balance_elapsed[:] = 6*dwell+.1
            p.before_action(state(32))
            torch.testing.assert_close(p.control.command[::2, 1], -p.control.command[1::2, 1])
            self.assertAlmostEqual(float(p.control.command[:, 1].abs().max()), .5, places=6)

    def test_faster_probes_keep_failure_reset_recovery_and_budget_gate(self):
        cfg = load_config(); c = Curriculum(cfg)
        self.assertTrue(c.probe_due(25))
        c.observe_probe(25, 26, 32, 0)
        c.observe_probe(50, 25, 32, 0)
        c.observe_probe(75, 26, 32, 0)
        restored = Curriculum(cfg); restored.load_state_dict(c.state_dict())
        self.assertEqual(restored.level, 0)
        self.assertFalse(restored.probe_due(75))
        self.assertEqual(restored.next_probe_update(75), 100)
        restored.observe_probe(100, 26, 32, 0)
        self.assertEqual(restored.level, 1)
        restored.level = cfg.max_course_level-1; restored.streak = 0
        restored.observe_probe(975, 32, 32, restored.level)
        final = restored.observe_probe(1000, 32, 32, restored.level)
        self.assertTrue(final['promotion_deferred_at_budget'])
        self.assertEqual(restored.level, cfg.max_course_level-1)

if __name__ == '__main__': unittest.main()
