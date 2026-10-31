"""Regression checks for the observed early-lift/oscillation reward exploit."""
import unittest
import torch
from step_demo.rewards import edge_potential, potential_difference, progress_increment, terminal_reward


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

    def test_v4_failure_does_not_get_cheaper_with_episode_length(self):
        failed=torch.ones(3,dtype=torch.bool);success=~failed
        r=terminal_reward(success,failed,torch.tensor([.14,1.22,15.]))
        torch.testing.assert_close(r,torch.full((3,),-5.))
        # The observed rear-hop moves <=2 cm and has no front clearance.
        p=edge_potential(torch.tensor([-.006,.019]),torch.full((2,2),.13),torch.zeros(2,2),.02)
        gain,_=progress_increment(torch.zeros(2),p,torch.zeros(2,dtype=torch.bool))
        self.assertEqual(float(gain.sum()),0.)

    def test_v4_progress_cannot_be_earned_twice_or_on_invalid_frame(self):
        best=torch.zeros(1);total=0.
        for value,invalid in [(1.5,False),(.2,False),(1.5,False),(2.,True),(1.5,False)]:
            gain,best=progress_increment(best,torch.tensor([value]),torch.tensor([invalid]));total+=float(gain)
        self.assertAlmostEqual(total,1.)
        far=edge_potential(torch.tensor([100.]),torch.zeros(1,2),torch.full((1,2),.03),.02)
        _,best=progress_increment(best,far,torch.tensor([False]));self.assertLessEqual(float(best),8.)

    def test_v4_task_return_orders_progress_and_completion(self):
        # Deterministic task-budget counterexamples, not simulated policies.
        # Each column is best valid potential, landing/ascent credit, motion cost.
        scenarios=[(0.,0.,0.),(0.,0.,.03),(1.7,0.,.1),(5.,4.,.4),(7.,7.,.8)]
        values=[]
        for i,(p,milestone,cost) in enumerate(scenarios):
            gain,_=progress_increment(torch.zeros(1),torch.tensor([p]),torch.tensor([False]))
            success=torch.tensor([i==4]);values.append(float(gain)+milestone-cost+float(
                terminal_reward(success,~success,torch.tensor([5.]))))
        wait,early,approach,landing,complete=values
        self.assertLess(early,wait);self.assertLess(wait,approach)
        self.assertLess(approach,landing);self.assertLess(landing,complete)


if __name__=='__main__':unittest.main()
