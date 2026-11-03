"""Keep airborne/other-stage authority and physical target constraints while braking."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config
from upright.controller import Controller, Phase
from upright.model import load_model
from checks.test_contracts import state

class GroundBraking(unittest.TestCase):
    def pair(self, n=1):
        cfg=load_config('configs/round21r3.json')
        return [Controller(n,'cpu',.03,replace(cfg,landing_ground_brake_slew_rad_s2=r),load_model(),.1) for r in (0.,450.)]

    def test_only_ground_land_changes(self):
        for phase in Phase:
            for gap in (0.,.013,.026,.10):
                pair=self.pair(32);s=state(32);s['gap'][:]=gap
                a=torch.rand(32,3,generator=torch.Generator().manual_seed(303))*2-1
                for c in pair:
                    c.phase[:]=phase;c.executed[:,1:]=-20.
                    c.set_command([[.6,.5]]*32)
                targets=[c.targets(a,s).clone() for c in pair]
                if phase!=Phase.LAND or gap>.012:
                    torch.testing.assert_close(*targets,rtol=0,atol=0)
                    self.assertFalse(pair[1].landing_ground_brake_delta.any())

    def test_faster_braking_without_sign_change_or_state_write(self):
        old,new=self.pair(2);s=state(2);motion=s['motion'].clone()
        for c in (old,new):c.phase[:]=Phase.LAND;c.executed[:,1:]=torch.tensor([[-20.,-20.],[20.,20.]])
        a=torch.zeros(2,3)
        torch.testing.assert_close(old.targets(a,s)[:,1:],torch.tensor([[-15.5,-15.5],[15.5,15.5]]))
        torch.testing.assert_close(new.targets(a,s)[:,1:],torch.tensor([[-6.5,-6.5],[6.5,6.5]]))
        new.targets(a,s)
        torch.testing.assert_close(new.executed[:,1:],torch.zeros(2,2))
        torch.testing.assert_close(s['motion'],motion)
        self.assertTrue((new.phase==Phase.LAND).all())

    def test_acceleration_and_live_handoff_unchanged(self):
        pair=self.pair(2);s=state(2);a=torch.zeros(2,3);a[:,1]=torch.tensor([-1.,1.])
        for c in pair:
            c.phase[:]=Phase.LAND;c.transition_motion[:,0]=torch.tensor([-.75,.75]);c.executed[:,1:]=torch.tensor([[-10.,-10.],[10.,10.]])
        torch.testing.assert_close(*(c.targets(a,s).clone() for c in pair),rtol=0,atol=0)
        for c in pair:
            c.phase[:]=Phase.HANDOFF;c.set_command([[.6,0],[-.6,0]])
            torch.testing.assert_close(c.targets(a,s)[:,1:],torch.tensor([[20.,20.],[-20.,-20.]]))

    def test_invalid_configuration(self):
        for rate in [-1,True,100.,1001.,float('nan'),'450']:
            with self.assertRaises(ValueError):load_config('configs/round21r3.json',landing_ground_brake_slew_rad_s2=rate)
        with self.assertRaises(ValueError):load_config('configs/round21r3.json',landing_projection_start_gap_m=0.,landing_ground_brake_slew_rad_s2=450.)

if __name__=='__main__':unittest.main()
