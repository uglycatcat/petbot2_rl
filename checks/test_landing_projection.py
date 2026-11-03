"""LAND target envelope, unchanged ownership, and source-limited actuator slew."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config
from upright.controller import Controller, Phase
from upright.model import load_model
from checks.test_contracts import state

class LandingProjection(unittest.TestCase):
    def pair(self,n=3):
        cfg=load_config('configs/round21r2_control.json')
        return [Controller(n,'cpu',.03,replace(cfg,landing_projection_start_gap_m=g),load_model(),.1) for g in (0.,.04)]

    def test_other_phases_exact_and_no_ordinary_range_reduction(self):
        for phase in (Phase.ORDINARY,Phase.RISE,Phase.BALANCE,Phase.HANDOFF,Phase.EXIT):
            pair=self.pair();s=state(3)
            for c in pair:
                c.phase[:]=phase;c.set_command([[.75,.5],[-.75,-.5],[0,0]])
                c.executed[:]=torch.tensor([.4,-20.,-20.]);c.handoff_start.copy_(c.executed)
            for _ in range(7):
                targets=[c.targets(torch.tensor([[.2,-.9,.5]]).repeat(3,1),s).clone() for c in pair]
                torch.testing.assert_close(*targets,rtol=0,atol=0)
                self.assertFalse(pair[1].landing_projection_weight.any())

    def test_zero_forward_backward_interval_and_raw_request(self):
        _,c=self.pair();s=state(3);c.phase[:]=Phase.LAND;c.transition_motion[:,0]=torch.tensor([0.,.3,-.3])
        action=torch.tensor([[0.,-1.,0.],[0.,-1.,0.],[0.,1.,0.]])
        before=s['motion'].clone()
        c.targets(action,s)
        torch.testing.assert_close(c.landing_projected_common,torch.zeros(3))
        torch.testing.assert_close(c.requested_action,action)
        self.assertTrue((c.phase==Phase.LAND).all());torch.testing.assert_close(s['motion'],before)
        action[:,1]=torch.tensor([1.,1.,-1.]);c.targets(action,s)
        torch.testing.assert_close(c.landing_projected_common,torch.tensor([0.,10.,-10.]))

    def test_full_air_authority_and_continuous_blend_with_slew(self):
        pair=self.pair();s=state(3);s['gap'][:]=.04
        for c in pair:c.phase[:]=Phase.LAND;c.executed[:,1:]=-20.
        action=torch.zeros(3,3);action[:,1]=-1.
        torch.testing.assert_close(*(c.targets(action,s).clone() for c in pair),rtol=0,atol=0)
        c=pair[1];s['gap'][:]=.026;c.targets(action,s)
        torch.testing.assert_close(c.landing_projection_weight,torch.full((3,),.5))
        torch.testing.assert_close(c.landing_projected_common,torch.full((3,),-12.5))
        previous=c.executed.clone();s['gap'][:]=0.;c.targets(action,s)
        self.assertLessEqual(float((c.executed[:,1:]-previous[:,1:]).abs().max()),4.50001)
        self.assertTrue((c.phase==Phase.LAND).all())

    def test_handoff_still_waits_and_live_input_owns_ground_motion(self):
        _,c=self.pair();s=state(3);c.phase[:]=Phase.LAND
        c.set_command([[0,0],[.6,0],[-.6,0]])
        for _ in range(6):c.update(s)
        self.assertTrue((c.phase==Phase.LAND).all())
        c.update(s);self.assertTrue((c.phase==Phase.HANDOFF).all())
        torch.testing.assert_close(c.targets(torch.ones(3,3),s)[:,1:],torch.tensor([[0.,0.],[20.,20.],[-20.,-20.]]))

    def test_invalid_range_rejected(self):
        for gap in [-1,.012,.08,True,float('nan'),'0.04']:
            with self.assertRaises(ValueError):load_config('configs/round21r2_control.json',landing_projection_start_gap_m=gap)

if __name__=='__main__':unittest.main()
