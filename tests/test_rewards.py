"""Regression checks for the observed early-lift/oscillation reward exploit."""
import unittest
import torch
from step_demo.rewards import edge_potential, potential_difference


class EdgeRewardTests(unittest.TestCase):
    def test_far_lift_has_no_credit_but_near_clearance_does(self):
        x=torch.zeros(2);gap=torch.tensor([[.14,.14],[.02,.02]])
        up=edge_potential(x,gap,torch.full((2,2),.03),.02)
        down=edge_potential(x,gap,torch.zeros(2,2),.02)
        self.assertEqual(float(up[0]),float(down[0]))
        self.assertGreater(float(up[1]),float(down[1]))

    def test_discounted_oscillation_cannot_farm_reward(self):
        gamma=.995;previous=torch.tensor([0.]);total=0.
        for t,value in enumerate([1.,2.,1.,0.,2.,1.,0.]):
            r,previous=potential_difference(previous,torch.tensor([value]),gamma,torch.tensor([t==6]))
            total+=gamma**t*float(r)
        self.assertAlmostEqual(total,0.,places=6)

    def test_holding_and_overshooting_do_not_earn_progress(self):
        p=edge_potential(torch.tensor([.65,2.]),torch.zeros(2,2),torch.full((2,2),.03),.02)
        torch.testing.assert_close(p[0],p[1])
        r,_=potential_difference(p,p,.995,torch.zeros(2,dtype=torch.bool))
        self.assertTrue(bool((r<0).all()))
        ended,next_p=potential_difference(p,p,.995,torch.ones(2,dtype=torch.bool))
        torch.testing.assert_close(ended,-p)
        self.assertEqual(float(next_p.sum()),0.)


if __name__=='__main__':unittest.main()
