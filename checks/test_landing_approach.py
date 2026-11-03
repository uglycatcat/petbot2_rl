"""Near-ground speed credit must not demand exact speed or alter other phases."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config
from upright.runtime import Runtime
from upright.controller import Phase
from upright.model import load_model
from checks.test_contracts import state, truth


class LandingApproach(unittest.TestCase):
    def runtime(self, n=6):
        cfg = load_config('configs/round20.json', num_envs=32)
        return Runtime(n, 'cpu', .03, replace(cfg, normal_handoff_live_drive=True,
                       landing_approach_motion_cost=2.), load_model())

    def test_interval_allows_slowing_in_both_directions(self):
        p = self.runtime(); s, t = state(6), truth(6)
        p.control.transition_motion[:, 0] = torch.tensor([.6, .6, -.6, -.6, 0., 0.])
        s['motion'][:, 0] = torch.tensor([.6, 0., -.3, 0., .02, -.02])
        cost = p.landing_approach_cost(s, t, torch.ones(6, dtype=torch.bool), torch.ones(6))
        self.assertFalse(cost.any())

    def test_symmetric_bounded_reverse_and_overspeed_cost(self):
        p = self.runtime(); s, t = state(6), truth(6)
        p.control.transition_motion[:, 0] = torch.tensor([.3, -.3, .3, -.3, 0., 0.])
        s['motion'][:, 0] = torch.tensor([-.2, .2, .5, -.5, -.2, .2])
        cost = p.landing_approach_cost(s, t, torch.ones(6, dtype=torch.bool), torch.ones(6))
        self.assertTrue((cost > 0).all() and (cost < .03*2).all())
        torch.testing.assert_close(cost[::2], cost[1::2])
        t['front_z'][:] = .15
        self.assertFalse(p.landing_approach_cost(s, t, torch.ones(6, dtype=torch.bool), torch.ones(6)).any())

    def test_other_stage_rewards_identical(self):
        for phase in (Phase.ORDINARY, Phase.RISE, Phase.BALANCE, Phase.HANDOFF, Phase.EXIT):
            original, candidate = self.runtime(), self.runtime()
            original.cfg = replace(original.cfg, landing_approach_motion_cost=0.)
            s, t = state(6), truth(6); s['motion'][:, 0] = -.4
            for p in (original, candidate): p.control.phase[:] = phase
            a, b = original.advance(s, t), candidate.advance(s, t)
            torch.testing.assert_close(a['reward'], b['reward'], rtol=0, atol=0)
            self.assertFalse(b['landing_approach_motion_cost'].any())


if __name__ == '__main__': unittest.main()
