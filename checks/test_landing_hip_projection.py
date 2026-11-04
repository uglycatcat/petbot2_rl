"""Near-ground hip return must preserve air authority, motion and handoff ownership."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config
from upright.controller import Controller, Phase
from upright.model import load_model
from checks.test_contracts import state


class LandingHipProjection(unittest.TestCase):
    def pair(self, n=3):
        cfg = load_config('configs/round21r5.json')
        return [Controller(n, 'cpu', .03, replace(cfg, landing_hip_default_projection=b),
                           load_model(), .1) for b in (False, True)]

    def test_other_phases_and_air_authority_exact(self):
        for phase in Phase:
            old, new = self.pair(); s = state(3)
            s['gap'][:] = .05 if phase == Phase.LAND else 0.
            for c in (old, new):
                c.phase[:] = phase; c.set_command([[.75, .5], [-.75, -.5], [0, 0]], hip=.3)
                c.executed[:] = torch.tensor([.4, -20., -20.]); c.handoff_start.copy_(c.executed)
            for _ in range(8):
                a = torch.tensor([[.9, -.7, .5], [-.9, .7, -.5], [0, 0, 0.]])
                torch.testing.assert_close(old.targets(a, s), new.targets(a, s), rtol=0, atol=0)

    def test_partial_and_full_projection_preserve_wheel_targets_and_raw_actions(self):
        old, new = self.pair(); s = state(3)
        s['gap'][:] = torch.tensor([[.04, .04], [.026, .026], [0., 0.]])
        for c in (old, new):
            c.phase[:] = Phase.LAND; c.transition_motion[:, 0] = torch.tensor([0., .3, -.3])
        a = torch.tensor([[.8, 1, .4], [.8, 1, .4], [.8, -1, -.4]])
        motion = s['motion'].clone()
        for _ in range(10):
            x, y = old.targets(a, s).clone(), new.targets(a, s).clone()
            torch.testing.assert_close(x[:, 1:], y[:, 1:], rtol=0, atol=0)
        torch.testing.assert_close(y[:, 0], x[:, 0]*torch.tensor([1., .5, 0.]))
        torch.testing.assert_close(new.requested_action, a)
        torch.testing.assert_close(s['motion'], motion)
        self.assertTrue((new.phase == Phase.LAND).all())

    def test_near_ground_hip_slew_and_live_ordinary_handoff(self):
        _, c = self.pair(); s = state(3); c.phase[:] = Phase.LAND
        c.executed[:, 0] = torch.tensor([.8, -.8, .1])
        c.set_command([[0, 0], [.6, 0], [-.6, 0]])
        prev = c.executed.clone()
        y = c.targets(torch.ones(3, 3), s)
        self.assertLessEqual(float((y[:, 0]-prev[:, 0]).abs().max()), .18001)
        self.assertTrue((y[:, 0].abs() < prev[:, 0].abs()).all())
        for _ in range(6): c.update(s)
        self.assertTrue((c.phase == Phase.LAND).all())
        c.update(s); self.assertTrue((c.phase == Phase.HANDOFF).all())
        torch.testing.assert_close(c.targets(torch.ones(3, 3), s)[:, 1:],
                                   torch.tensor([[0., 0.], [20., 20.], [-20., -20.]]))

    def test_configuration_requires_envelope_and_explicit_boolean(self):
        cfg = load_config('configs/round21r5.json')
        self.assertFalse(cfg.landing_hip_default_projection)
        for value in (1, 0, 'true', None, float('nan')):
            with self.assertRaises(ValueError): replace(cfg, landing_hip_default_projection=value).validate()
        with self.assertRaises(ValueError):
            replace(cfg, landing_hip_default_projection=True, landing_projection_start_gap_m=0.).validate()


if __name__ == '__main__': unittest.main()
