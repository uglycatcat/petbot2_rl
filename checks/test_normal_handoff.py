"""Normal landing input ownership, with legacy/abnormal paths preserved."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config
from upright.controller import Controller, Phase
from upright.model import load_model
from checks.test_contracts import state


class NormalHandoff(unittest.TestCase):
    def pair(self):
        cfg = load_config('configs/round20.json', num_envs=32)
        return [Controller(3, 'cpu', .03, replace(cfg, normal_handoff_live_drive=flag),
                           load_model(), .1) for flag in (False, True)]

    def test_handoff_current_input_without_old_wheel_target(self):
        old, new = self.pair(); s = state(3)
        command = torch.tensor([[0., 0.], [.6, 0.], [-.6, 0.]])
        s['motion'][:, 0] = -.3
        s['q'][:, 3:] = .5
        for c in (old, new):
            c.phase[:] = Phase.BALANCE
            c.executed[:] = torch.tensor([.5, -24., -24.])
            c.set_command(command)
            c.request_land(s, torch.arange(3))
            for _ in range(7): c.update(s)
            self.assertTrue((c.phase == Phase.HANDOFF).all())
        torch.testing.assert_close(new.transition_motion[:, 0], torch.full((3,), -.3))
        torch.testing.assert_close(new.command, command)
        torch.testing.assert_close(old.command[:, 0], torch.full((3,), -.3))
        a, b = [c.targets(torch.zeros(3, 3), s).clone() for c in (old, new)]
        torch.testing.assert_close(b[:, 1:], torch.tensor([[0., 0.], [20., 20.], [-20., -20.]]))
        torch.testing.assert_close(a[:, 0], b[:, 0])
        # Input can change during the ordinary confirmation interval.
        new.set_command(-command)
        torch.testing.assert_close(new.targets(torch.ones(3, 3), s)[:, 1:], -b[:, 1:])
        self.assertFalse(new.completed.any())

    def test_policy_phases_and_abnormal_exit_unchanged(self):
        for phase in (Phase.RISE, Phase.BALANCE, Phase.LAND, Phase.EXIT):
            old, new = self.pair(); s = state(3)
            for c in (old, new):
                c.phase[:] = phase
                c.executed[:] = torch.tensor([.5, -24., -24.])
                c.handoff_start.copy_(c.executed)
                c.set_command([[.4, -.5], [0., 0.], [-.4, .5]])
            for _ in range(4):
                a, b = [c.targets(torch.tensor([[.2, -.3, .4]]).repeat(3, 1), s).clone()
                        for c in (old, new)]
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        old, new = self.pair(); s = state(3); s['g'][:, 2] = 1.; s['motion'][:, 0] = .2
        for c in (old, new):
            c.phase[:] = Phase.LAND; c.set_command([[0., 0.]]*3); c.update(s)
            self.assertTrue((c.phase == Phase.EXIT).all())
        torch.testing.assert_close(old.command, new.command)

    def test_no_early_handoff_and_no_in_air_zeroing(self):
        _, c = self.pair(); s = state(3); c.phase[:] = Phase.LAND
        s['gap'][:] = .04
        c.executed[:, 1:] = -20.
        for _ in range(10): c.update(s)
        self.assertTrue((c.phase == Phase.LAND).all())
        self.assertTrue((c.targets(torch.zeros(3, 3), s)[:, 1:] < 0).all())
        s['gap'][:] = 0.
        for _ in range(6): c.update(s)
        self.assertTrue((c.phase == Phase.LAND).all())
        c.update(s)
        self.assertTrue((c.phase == Phase.HANDOFF).all())


if __name__ == '__main__': unittest.main()
